"""Shared best-effort cleanup for resources owned by one data-factory run."""

from __future__ import annotations

import asyncio
from collections.abc import Collection
from contextlib import suppress

from codeworld.runtime.container import DockerRuntime


async def cleanup_owned_resources(
    runtime: DockerRuntime,
    *,
    protected_images: Collection[str] | None,
) -> None:
    """Remove exactly owned transient resources without masking the stage outcome."""

    with suppress(Exception):
        await runtime.cleanup_run_resources(protected_images=protected_images)


async def finish_owned_cleanup(
    runtime: DockerRuntime,
    *,
    protected_images: Collection[str] | None,
) -> None:
    """Finish bounded cleanup before propagating cancellation from the stage."""

    cleanup = asyncio.create_task(
        cleanup_owned_resources(runtime, protected_images=protected_images)
    )
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        with suppress(asyncio.CancelledError):
            await cleanup
        raise
