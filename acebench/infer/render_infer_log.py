"""Render OpenHands infer.log to human-friendly Markdown/HTML."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BLOCK_HEADER_RE = re.compile(r"^\d{2}:\d{2}:\d{2} - (USER_ACTION|ACTION|OBSERVATION)\b")
ACTION_NAME_RE = re.compile(r"(?:\]|\*\*|\b)([A-Za-z0-9_]+Action)\b")
OBS_NAME_RE = re.compile(r"\*\*([A-Za-z0-9_]+Observation)\*\*")
PREVIEW_LINE_COUNT = 3
ACTION_COLORS = {
    "filereadaction": "#3b82f6",
    "filewriteaction": "#10b981",
    "fileeditaction": "#22c55e",
    "cmdrunaction": "#f59e0b",
    "browseurlaction": "#a855f7",
    "browseinteractiveaction": "#8b5cf6",
    "ipythonruncellaction": "#0ea5e9",
    "agentfinishaction": "#14b8a6",
    "agentrejectaction": "#ef4444",
    "agentdelegateaction": "#f97316",
    "changeagentstateaction": "#eab308",
    "messageaction": "#84cc16",
    "systemmessageaction": "#6366f1",
    "agentthinkaction": "#64748b",
    "recallaction": "#06b6d4",
    "mcpaction": "#ec4899",
    "tasktrackingaction": "#d97706",
    "looprecoveryaction": "#ef4444",
    "nullaction": "#94a3b8",
}


@dataclass
class LogBlock:
    kind: str  # USER_ACTION | ACTION | OBSERVATION
    header: str
    lines: list[str]
    name: str | None = None  # Action/Observation class name


def _iter_blocks(lines: Iterable[str]) -> Iterable[LogBlock]:
    current: LogBlock | None = None
    for raw in lines:
        line = raw.rstrip("\n")
        m = BLOCK_HEADER_RE.match(line)
        if m:
            if current is not None:
                yield current
            kind = m.group(1)
            current = LogBlock(kind=kind, header=line, lines=[])
            continue
        if current is not None:
            current.lines.append(line)
    if current is not None:
        yield current


def _infer_name(block: LogBlock) -> None:
    if block.name:
        return
    joined = "\n".join(block.lines)
    if block.kind in {"ACTION", "USER_ACTION"}:
        m = ACTION_NAME_RE.search(joined)
        if m:
            block.name = m.group(1)
    elif block.kind == "OBSERVATION":
        m = OBS_NAME_RE.search(joined)
        if m:
            block.name = m.group(1)


def _escape(s: str) -> str:
    return html.escape(s, quote=False)


def _block_to_md(block: LogBlock) -> str:
    _infer_name(block)
    title = block.name or block.kind
    body = "\n".join(block.lines).strip("\n")
    return f"### {title}\n\n```\n{body}\n```\n"


def _block_to_html(block: LogBlock) -> str:
    _infer_name(block)
    name = block.name or ""
    raw_body = "\n".join(block.lines).strip("\n")
    body = _escape(raw_body)
    preview_lines = raw_body.splitlines()[:PREVIEW_LINE_COUNT]
    preview = _escape("\n".join(preview_lines))
    kind_cls = block.kind.lower()
    name_cls = (name or "unknown").lower()
    title = name or block.kind
    return (
        f"<details class='block {kind_cls} {name_cls}'>"
        f"<summary class='block-title'>"
        f"<div class='title-row'>"
        f"<span class='block-kind'>{_escape(title)}</span>"
        f"<span class='hint'>(点击展开)</span>"
        "</div>"
        f"<pre class='preview'>{preview}</pre>"
        "</summary>"
        f"<pre class='full'>{body}</pre>"
        "</details>"
    )


def _render_html(blocks: list[LogBlock], title: str) -> str:
    blocks_html = "\n".join(_block_to_html(b) for b in blocks)
    color_rules = "\n".join(
        f"    .block.{name} {{ --accent: {color}; }}"
        for name, color in ACTION_COLORS.items()
    )
    return (
        "<!doctype html>\n"
        "<html lang='en'>\n"
        "<head>\n"
        "  <meta charset='utf-8' />\n"
        "  <meta name='viewport' content='width=device-width, initial-scale=1' />\n"
        f"  <title>{_escape(title)}</title>\n"
        "  <style>\n"
        "    :root {\n"
        "      --bg: #0f1115;\n"
        "      --panel: #161a22;\n"
        "      --text: #e6e6e6;\n"
        "      --muted: #9aa4b2;\n"
        "      --action: #3b82f6;\n"
        "      --user: #22c55e;\n"
        "      --obs: #f59e0b;\n"
        "      --accent: #2f3747;\n"
        "    }\n"
        "    * { box-sizing: border-box; }\n"
        "    body { margin: 0; padding: 24px; background: var(--bg); color: var(--text); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }\n"
        "    h1 { font-size: 20px; margin: 0 0 16px 0; color: var(--text); }\n"
        "    .block { background: var(--panel); border: 1px solid #232a36; border-left: 4px solid var(--accent); border-radius: 10px; padding: 12px 14px; margin: 12px 0; }\n"
        "    .block-title { font-weight: 700; margin-bottom: 8px; }\n"
        "    summary.block-title { display: flex; flex-direction: column; gap: 6px; cursor: pointer; list-style: none; }\n"
        "    .title-row { display: flex; justify-content: space-between; align-items: center; gap: 12px; }\n"
        "    summary.block-title::-webkit-details-marker { display: none; }\n"
        "    .hint { color: var(--muted); font-weight: 400; font-size: 12px; }\n"
        "    .block.action { --accent: var(--action); }\n"
        "    .block.user_action { --accent: var(--user); }\n"
        "    .block.observation { --accent: var(--obs); }\n"
        "    .block-kind { color: var(--accent); }\n"
        "    .preview { opacity: 0.9; }\n"
        "    details[open] summary .preview { display: none; }\n"
        "    details[open] .full { display: block; }\n"
        "    .full { display: none; }\n"
        "    pre { margin: 0; white-space: pre-wrap; word-break: break-word; color: var(--text); font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12.5px; line-height: 1.5; }\n"
        "    .meta { color: var(--muted); font-size: 12px; margin-top: 4px; }\n"
        f"{color_rules}\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"  <h1>{_escape(title)}</h1>\n"
        f"  {blocks_html}\n"
        "</body>\n"
        "</html>\n"
    )


def render_infer_log(log_path: Path, mode: str) -> tuple[str, str]:
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    blocks = list(_iter_blocks(lines))

    if mode == "full":
        keep = {"USER_ACTION", "ACTION", "OBSERVATION"}
    else:
        keep = {"USER_ACTION", "ACTION"}

    blocks = [b for b in blocks if b.kind in keep]

    md = "\n".join(_block_to_md(b) for b in blocks)
    html_doc = _render_html(blocks, title=str(log_path))
    return md, html_doc
