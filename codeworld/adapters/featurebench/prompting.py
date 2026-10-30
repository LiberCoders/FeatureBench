"""Official-style Level 1 issue authoring and interface rendering."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from codeworld.adapters.featurebench.indexing import RepositoryIndex
from codeworld.adapters.featurebench.models import (
    InterfaceDescription,
    IssueDraft,
    MaskPlan,
    SourceObject,
)
from codeworld.errors import ContractError

_TEMPLATE_DIRECTORY = Path(__file__).resolve().parent / "assets"


def _empty_children() -> dict[str, _InterfaceNode]:
    return {}


@dataclass(slots=True)
class _InterfaceNode:
    source: SourceObject
    children: dict[str, _InterfaceNode] = field(default_factory=_empty_children)
    contract: str | None = None


def _render_docstring(docstring: str, indentation: str) -> list[str]:
    if not docstring:
        return []
    clean = docstring.replace('"""', r"\"\"\"")
    if "\n" not in clean:
        return [f'{indentation}"""{clean}"""']
    return [
        f'{indentation}"""',
        *(f"{indentation}{line}" for line in clean.splitlines()),
        f'{indentation}"""',
    ]


def _render_node(node: _InterfaceNode, indent: int) -> list[str]:
    prefix = "    " * indent
    lines: list[str] = []
    for decorator in node.source.decorators:
        lines.append(f"{prefix}@{decorator.removeprefix('@')}")
    signature = textwrap.dedent(node.source.signature).strip()
    lines.extend(textwrap.indent(signature, prefix).splitlines())
    body_prefix = prefix + "    "
    if node.contract is not None:
        lines.extend(_render_docstring(node.contract, body_prefix))
    if node.children:
        if node.contract is not None:
            lines.append("")
        for index, child in enumerate(node.children.values()):
            if index:
                lines.append("")
            lines.extend(_render_node(child, indent + 1))
    else:
        lines.append(f"{body_prefix}# <your code>")
    return lines


def build_interfaces(
    index: RepositoryIndex,
    plan: MaskPlan,
    draft: IssueDraft,
) -> tuple[InterfaceDescription, ...]:
    """Render final Top objects and only the minimal ancestor syntax needed to locate them."""
    contracts = {item.object_id: item.docstring for item in draft.interface_contracts}
    expected = {item.object_id for item in plan.top_objects}
    if set(contracts) != expected:
        raise ContractError("Issue contracts must describe every and only final Top object")
    roots_by_path: dict[str, dict[str, _InterfaceNode]] = {}
    for target in plan.top_objects:
        chain = (*index.ancestors(target), target)
        current = roots_by_path.setdefault(target.path, {})
        for item in chain:
            node = current.setdefault(item.object_id, _InterfaceNode(source=item))
            if item.object_id == target.object_id:
                node.contract = contracts[target.object_id]
            current = node.children
    interfaces: list[InterfaceDescription] = []
    for number, (path, roots) in enumerate(sorted(roots_by_path.items()), start=1):
        code_lines: list[str] = []
        for position, node in enumerate(roots.values()):
            if position:
                code_lines.append("")
            code_lines.extend(_render_node(node, 0))
        interfaces.append(
            InterfaceDescription(
                path="/testbed/" + path.removeprefix("/testbed/").lstrip("/"),
                code="\n".join(code_lines).strip(),
                description=f"Below is **Interface Description {number}**",
            )
        )
    return tuple(interfaces)


def render_prompt(
    *,
    draft: IssueDraft,
    interfaces: tuple[InterfaceDescription, ...],
    library_name: str,
    black_links: tuple[str, ...],
) -> str:
    if not interfaces:
        raise ContractError("FeatureBench prompt requires at least one interface description")
    first_code = interfaces[0].code
    marker = first_code.find("<your code>")
    if marker >= 0:
        interface_example = first_code[: marker + len("<your code>")] + "\n..."
    else:
        interface_example = first_code[:200] + ("\n..." if len(first_code) > 200 else "")
    rendered_black_links = (
        "black_links:\n" + "\n".join(f"- {link}" for link in black_links) if black_links else ""
    )
    variables: dict[str, object] = {
        "task_statement": draft.task_statement,
        "task_name": "",
        "library_name": library_name,
        "black_links": rendered_black_links,
        "interface_code_example_path": interfaces[0].path,
        "interface_code_example": interface_example,
    }
    for number, interface in enumerate(interfaces, start=1):
        variables[f"interface_description{number}"] = interface.description
        variables[f"interface_code{number}"] = interface.code
        variables[f"interface_code_path{number}"] = interface.path
    variables["all_vars"] = variables
    environment = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIRECTORY),
        autoescape=select_autoescape(default=False),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    return environment.get_template("level_1.md.j2").render(variables).strip()
