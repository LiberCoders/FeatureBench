"""Joblib Backend Patch - force joblib to run tasks in the current process for tracing."""
from __future__ import annotations

import os
import types
from typing import Any, Optional


class JoblibBackendPatcher:
    """Patch joblib.Parallel to avoid spawning subprocesses."""

    def __init__(self, tracer: Any) -> None:
        self.tracer = tracer
        self._original_init: Optional[types.FunctionType] = None
        self._hooked: bool = False

    def hook(self) -> None:
        if self._hooked:
            return

        # Disable joblib multiprocessing backend to avoid untraceable subprocesses
        os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

        try:
            import joblib  # noqa: F401  # pylint: disable=unused-import
            from joblib import parallel
        except Exception as exc:  # pragma: no cover - joblib unavailable in runtime
            if self.tracer.debug:
                print(f"JoblibBackendPatcher: failed to import joblib: {exc}")
            return

        if self._original_init is None:
            self._original_init = parallel.Parallel.__init__

        tracer_debug = self.tracer.debug

        def patched_init(self_parallel, *args, **kwargs):  # type: ignore[override]
            # Copy kwargs to avoid mutating caller-provided dict
            kwargs = dict(kwargs)

            prefer = kwargs.get("prefer")
            if prefer is None or prefer == "processes":
                kwargs["prefer"] = "threads"
            elif isinstance(prefer, str) and "process" in prefer:
                kwargs["prefer"] = "threads"

            # Ensure at least one worker thread (current process)
            kwargs["n_jobs"] = max(1, int(kwargs.get("n_jobs", 1)))

            if tracer_debug:
                backend = kwargs.get("backend") or kwargs.get("prefer")
                print(f"JoblibBackendPatcher: forcing thread backend ({backend})")

            return self._original_init(self_parallel, *args, **kwargs)

        parallel.Parallel.__init__ = patched_init  # type: ignore[assignment]
        self._hooked = True

        if self.tracer.debug:
            print("✓ Joblib Backend Hook installed (thread backend)")

    def unhook(self) -> None:
        if not self._hooked:
            return

        try:
            from joblib import parallel
        except Exception:  # pragma: no cover
            return

        if self._original_init is not None:
            parallel.Parallel.__init__ = self._original_init  # type: ignore[assignment]

        self._hooked = False

        if self.tracer.debug:
            print("✓ Joblib Backend Hook uninstalled")
