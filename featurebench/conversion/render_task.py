import yaml
import os
from pathlib import Path
from typing import Dict
from jinja2 import Environment, FileSystemLoader, StrictUndefined


def find_project_root() -> Path:
    """Find the project root directory."""
    start = Path(__file__).resolve().parent
    # Walk current directory and parents
    for p in [start] + list(start.parents):
        if (p / ".git").exists():
            return p
    # If .git is missing, fall back to two levels up
    return start.parents[1]


def get_task(config_path: str) -> Dict[str, str]:
    """
    Read config.yaml, render templates, and return a task dict.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Dict[str, str]: Dictionary with keys:
            - "prompt": Rendered prompt.md content
            - "docs": technical_docs file paths (comma-separated, prefixed)
            - "black_links": Rendered black_links.txt content
    """
    # Convert to Path
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Resolve project root
    project_root = find_project_root()
    templates_dir = project_root / "featurebench" / "resources" / "templates"
    
    # Read config file
    with cfg_path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # Initialize Jinja2 environment
    env = Environment(
        loader=FileSystemLoader([templates_dir]),
        undefined=StrictUndefined,
    )
    env.globals["project_root"] = str(project_root)
    
    # Select prompt template based on task_level
    task_level = cfg.get("task_level", 1)
    prompt_template_name = f"level_{task_level}.j2"
    
    # Add all_vars for template iteration
    cfg["all_vars"] = cfg
    
    result = {}
    
    # 1. Build black_links content
    try:
        black_links = cfg.get("black_links", [])
        if black_links:
            black_links_content = ["black_links:"]
            for link in black_links:
                black_links_content.append(f"- {link}")
            all_black_links = "\n".join(black_links_content)
        else:
            all_black_links = ""
    except Exception as e:
        raise RuntimeError(f"Failed to generate black_links content: {e}")

    cfg["black_links"] = all_black_links
    # 2. Render prompt.md
    try:
        prompt_tpl = env.get_template(prompt_template_name)
        result["prompt"] = prompt_tpl.render(cfg)
    except Exception as e:
        raise RuntimeError(f"Failed to render prompt template {prompt_template_name}: {e}")
    
    return result