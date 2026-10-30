"""Harness FileSystem workspace assets for FeatureBench Issue authoring."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from codeworld.adapters.featurebench.indexing import RepositoryIndex
from codeworld.adapters.featurebench.models import MaskPlan
from codeworld.agents.feedback import FeedbackIssue
from codeworld.errors import ContractError

_ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "issue"


def _safe_relative_path(path: str) -> PurePosixPath:
    normalized = path.removeprefix("/testbed/").lstrip("/")
    relative = PurePosixPath(normalized)
    if not relative.parts or ".." in relative.parts or relative.is_absolute():
        raise ContractError(f"invalid repository source path for Issue workspace: {path!r}")
    return relative


@dataclass(slots=True)
class IssueAuthorWorkspace:
    """A real, sandboxed directory consumed through Harness's FileSystem capability."""

    root: Path

    protocol_path: str = "protocol/README.md"
    manifest_path: str = "input/top-interfaces.jsonl"
    draft_path: str = "outbox/issue.json"

    def prepare(self, *, index: RepositoryIndex, plan: MaskPlan) -> None:
        """Write immutable evidence copies and one incrementally editable draft."""
        protocol = self.root / "protocol"
        inputs = self.root / "input"
        sources = self.root / "sources"
        outbox = self.root / "outbox"
        for directory in (protocol, inputs, sources, outbox):
            directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_ASSET_ROOT / "protocol.md", protocol / "README.md")

        source_paths = sorted({item.path for item in plan.top_objects})
        for path in source_paths:
            try:
                content = index.sources[path]
            except KeyError as exc:
                raise ContractError(f"final interface source file is missing: {path}") from exc
            relative = _safe_relative_path(path)
            destination = sources.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")

        records: list[dict[str, object]] = []
        for item in plan.top_objects:
            relative = _safe_relative_path(item.path)
            records.append(
                {
                    "object_id": item.object_id,
                    "path": item.path,
                    "source_file": str(PurePosixPath("sources") / relative),
                    "qualified_name": item.qualname,
                    "kind": item.kind,
                    "signature": item.signature,
                    "decorators": item.decorators,
                    "existing_docstring": item.docstring,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                }
            )
        (inputs / "top-interfaces.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        (outbox / "issue.json").write_text(
            json.dumps(
                {
                    "task_statement": "",
                    "interface_contracts": [
                        {"object_id": item.object_id, "docstring": ""} for item in plan.top_objects
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def outbox_host_path(self, submitted_path: str) -> Path:
        """Resolve one model-supplied relative path strictly beneath outbox/."""
        raw = PurePosixPath(submitted_path)
        if raw.is_absolute() or ".." in raw.parts or not raw.parts or raw.parts[0] != "outbox":
            raise ValueError("submitted path must be relative and located under outbox/")
        resolved = (self.root / Path(*raw.parts)).resolve()
        outbox = (self.root / "outbox").resolve()
        if not resolved.is_relative_to(outbox):
            raise ValueError("submitted path resolves outside outbox/")
        return resolved

    def feedback_path(self, attempt_id: str) -> str:
        return str(PurePosixPath("attempts") / attempt_id / "feedback")

    def write_feedback(
        self,
        *,
        attempt_id: str,
        accepted: bool,
        issues: Sequence[FeedbackIssue],
    ) -> str:
        """Persist concise, field-addressable feedback for one submit."""
        relative = self.feedback_path(attempt_id)
        destination = self.root / Path(*PurePosixPath(relative).parts)
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "result.json").write_text(
            json.dumps(
                {
                    "attempt_id": attempt_id,
                    "accepted": accepted,
                    "violation_count": len(issues),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (destination / "violations.jsonl").write_text(
            "".join(
                json.dumps(item.model_dump(mode="json", exclude_none=True), ensure_ascii=False)
                + "\n"
                for item in issues
            ),
            encoding="utf-8",
        )
        readme = f"# Issue submit {attempt_id}\n\nAccepted: `{'true' if accepted else 'false'}`\n\n"
        if accepted:
            readme += "The submitted Issue file passed every host-owned contract.\n"
        else:
            readme += (
                "Read `violations.jsonl`, edit only the affected parts of `outbox/issue.json`, "
                "then submit that path again. Preserve already valid content.\n"
            )
        (destination / "README.md").write_text(readme, encoding="utf-8")
        return relative
