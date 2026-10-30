from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from codeworld.config import DiscoverySettings
from codeworld.discovery.github import GitHubDiscovery
from codeworld.discovery.run import DiscoveryRun
from codeworld.repositories.jsonl import RepositoryJsonlAppender, iter_repositories
from codeworld.state import StateStore

pytestmark = pytest.mark.integration
_PROJECT_ENV = Path(__file__).resolve().parents[2] / ".env"


@pytest.mark.asyncio
async def test_real_target_discovery_checkpoints_to_jsonl(tmp_path: Path) -> None:
    if os.getenv("FEATUREBENCH_RUN_NETWORK_TESTS") != "1":
        pytest.skip("set FEATUREBENCH_RUN_NETWORK_TESTS=1 for the real GitHub API smoke test")
    load_dotenv(_PROJECT_ENV, override=False)
    run_id = "github-smoke"
    output = tmp_path / "repositories.jsonl"
    state = StateStore(tmp_path / "state.sqlite3")
    state.create_run(run_id, "discover", {})
    token = os.getenv("GITHUB_TOKEN", "").split(",", 1)[0].strip()
    settings = DiscoverySettings(
        target_repositories=("octocat/Hello-World",),
        maximum_repositories=1,
        partition_concurrency=1,
        max_concurrent_requests=1,
        request_attempts=3,
    )
    try:
        async with GitHubDiscovery(
            settings,
            tokens=(token,) if token else (),
            run_id=run_id,
        ) as discovery:
            with RepositoryJsonlAppender(output, fsync_interval=1) as writer:
                total, appended = await DiscoveryRun(
                    discovery=discovery,
                    writer=writer,
                    state=state,
                    run_id=run_id,
                    output=output,
                    resume=False,
                ).run()
        records = list(iter_repositories(output))
        assert total == appended == 1
        assert len(records) == 1
        assert records[0].full_name == "octocat/Hello-World"
        assert records[0].discovery_run_id == run_id
        job = state.get_job(run_id, "target:00000000")
        assert job is not None
        assert job.status == "completed"
    finally:
        state.close()
