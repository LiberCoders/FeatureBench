from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from codeworld.config import DiscoverySettings
from codeworld.discovery.github import GitHubDiscovery
from codeworld.discovery.models import SearchPartition
from codeworld.discovery.run import DiscoveryRun
from codeworld.events import Event
from codeworld.repositories.jsonl import RepositoryJsonlAppender, iter_repositories
from codeworld.repositories.models import RepositoryCandidate
from codeworld.selection import default_work_selection_registry
from codeworld.state import StateStore


def candidate(run_id: str, index: int) -> RepositoryCandidate:
    timestamp = datetime(2025, 1, 1, tzinfo=UTC)
    return RepositoryCandidate(
        discovery_run_id=run_id,
        github_id=index,
        full_name=f"owner/repo-{index}",
        name=f"repo-{index}",
        owner="owner",
        clone_url=f"https://github.com/owner/repo-{index}.git",
        html_url=f"https://github.com/owner/repo-{index}",
        api_url=f"https://api.github.com/repos/owner/repo-{index}",
        default_branch="main",
        stars=10,
        forks=0,
        open_issues=0,
        size_kb=10,
        language="Python",
        created_at=timestamp,
        updated_at=timestamp,
    )


def github_item(
    index: int,
    *,
    language: str,
    updated_at: datetime,
    stars: int = 10,
) -> dict[str, Any]:
    full_name = f"owner/{language.casefold()}-{index}"
    timestamp = updated_at.isoformat().replace("+00:00", "Z")
    return {
        "id": index,
        "node_id": f"node-{index}",
        "full_name": full_name,
        "name": full_name.split("/", 1)[1],
        "owner": {"login": "owner"},
        "clone_url": f"https://github.com/{full_name}.git",
        "html_url": f"https://github.com/{full_name}",
        "url": f"https://api.github.com/repos/{full_name}",
        "default_branch": "main",
        "stargazers_count": stars,
        "forks_count": 0,
        "open_issues_count": 0,
        "size": 10,
        "language": language,
        "license": None,
        "topics": [],
        "visibility": "public",
        "archived": False,
        "fork": False,
        "private": False,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": timestamp,
        "pushed_at": timestamp,
    }


class FakeTargetDiscovery:
    def __init__(
        self,
        run_id: str,
        *,
        fail_on: str | None,
        configured_targets: tuple[str, ...] = (
            "owner/repo-1",
            "owner/repo-2",
            "owner/repo-3",
        ),
    ) -> None:
        self.run_id = run_id
        self.fail_on = fail_on
        self.calls: list[str] = []
        self.settings = DiscoverySettings(
            target_repositories=configured_targets,
            maximum_repositories=3,
            partition_concurrency=1,
        )

    async def fetch_target(self, full_name: str) -> RepositoryCandidate:
        self.calls.append(full_name)
        if full_name == self.fail_on:
            raise RuntimeError("simulated interruption")
        return candidate(self.run_id, int(full_name.rsplit("-", 1)[1]))


class RecordingEventSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)


class EmptySearchDiscovery(GitHubDiscovery):
    async def _count(self, partition: SearchPartition) -> int:
        del partition
        return 0


class StubBoundedSearchDiscovery(GitHubDiscovery):
    def __init__(
        self,
        settings: DiscoverySettings,
        items: dict[str, list[dict[str, Any]]],
    ) -> None:
        super().__init__(settings, run_id="bounded-run")
        self.items = items
        self.calls: list[tuple[str, int, int]] = []

    async def _search(
        self,
        query: str,
        *,
        page: int,
        per_page: int,
    ) -> dict[str, Any]:
        self.calls.append((query, page, per_page))
        language = next(name for name in self.items if f"language:{name}" in query)
        items = self.items[language]
        start = (page - 1) * per_page
        return {
            "total_count": len(items),
            "incomplete_results": False,
            "items": items[start : start + per_page],
        }


class FakeBoundedDiscovery:
    def __init__(self, run_id: str, maximum: int = 2) -> None:
        self.settings = DiscoverySettings(
            languages=("Python", "Go"),
            maximum_repositories=maximum,
            partition_concurrency=1,
        )
        self.run_id = run_id
        self.bounded_calls = 0

    async def fetch_bounded(self) -> list[RepositoryCandidate]:
        self.bounded_calls += 1
        maximum = self.settings.maximum_repositories
        assert maximum is not None
        return [candidate(self.run_id, index) for index in range(1, maximum + 1)]

    async def partitions(self, **_: object) -> list[SearchPartition]:
        raise AssertionError("bounded discovery must not plan lossless partitions")


class FakeLosslessDiscovery:
    def __init__(self, run_id: str) -> None:
        self.settings = DiscoverySettings(
            languages=("Python",),
            maximum_repositories=None,
            created_after=datetime(2025, 1, 1, tzinfo=UTC),
            created_before=datetime(2025, 1, 2, tzinfo=UTC),
            partition_concurrency=1,
        )
        self.run_id = run_id
        self.partition_calls = 0

    async def partitions(self, **_: object) -> list[SearchPartition]:
        self.partition_calls += 1
        return [
            SearchPartition(
                language="Python",
                created_start=self.settings.created_after,
                created_end=cast(datetime, self.settings.created_before),
                expected_count=1,
            )
        ]

    async def fetch_partition(
        self,
        index: int,
        partition: SearchPartition,
    ) -> list[RepositoryCandidate]:
        assert index == 0
        assert partition.language == "Python"
        return [candidate(self.run_id, 1)]


@pytest.mark.asyncio
async def test_discovery_planning_progress_tracks_resolved_search_coverage() -> None:
    events = RecordingEventSink()
    settings = DiscoverySettings(
        languages=("Python", "Rust"),
        created_after=datetime(2025, 1, 1, tzinfo=UTC),
        created_before=datetime(2025, 1, 2, tzinfo=UTC),
        partition_concurrency=1,
    )

    async with EmptySearchDiscovery(settings, events=events) as discovery:
        assert await discovery.partitions() == []

    progress = [event.data for event in events.events if event.kind == "discovery_progress"]
    assert [item["progress"] for item in progress] == [0.0, 0.5, 1.0]
    assert [item["current_language"] for item in progress] == [
        "Python",
        "Rust",
        "Rust",
    ]
    assert all(item["search_group_total"] == 2 for item in progress)


@pytest.mark.asyncio
async def test_bounded_search_fetches_per_group_top_k_and_merges_globally() -> None:
    settings = DiscoverySettings(
        languages=("Python", "Go"),
        created_after=datetime(2024, 1, 1, tzinfo=UTC),
        created_before=datetime(2025, 1, 1, tzinfo=UTC),
        maximum_repositories=2,
        requests_per_page=100,
        sort="updated",
        order="desc",
    )
    items = {
        "Python": [
            github_item(
                1,
                language="Python",
                updated_at=datetime(2025, 1, 4, tzinfo=UTC),
            ),
            github_item(
                2,
                language="Python",
                updated_at=datetime(2025, 1, 1, tzinfo=UTC),
            ),
        ],
        "Go": [
            github_item(
                3,
                language="Go",
                updated_at=datetime(2025, 1, 3, tzinfo=UTC),
            ),
            github_item(
                4,
                language="Go",
                updated_at=datetime(2025, 1, 2, tzinfo=UTC),
            ),
        ],
    }

    discovery = StubBoundedSearchDiscovery(settings, items)
    async with discovery:
        selected = await discovery.fetch_bounded()

    assert [item.github_id for item in selected] == [1, 3]
    assert len(discovery.calls) == 2
    assert {page for _, page, _ in discovery.calls} == {1}
    assert {per_page for _, _, per_page in discovery.calls} == {2}
    assert all(
        "created:2024-01-01T00:00:00Z..2025-01-01T00:00:00Z" in query
        for query, _, _ in discovery.calls
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("selection_name", ("unseen", "non-success"))
async def test_discovery_run_selects_unseen_bounded_job_without_partition_planning(
    tmp_path: Path,
    selection_name: str,
) -> None:
    run_id = "bounded-run"
    output = tmp_path / "repositories.jsonl"
    state = StateStore(tmp_path / "state.sqlite3")
    state.create_run(run_id, "discover", {})
    discovery = FakeBoundedDiscovery(run_id)

    with RepositoryJsonlAppender(output, fsync_interval=1) as writer:
        total, appended = await DiscoveryRun(
            discovery=cast(GitHubDiscovery, discovery),
            writer=writer,
            state=state,
            run_id=run_id,
            output=output,
            resume=False,
            selection=default_work_selection_registry().resolve(selection_name),
        ).run()

    assert total == appended == 2
    assert discovery.bounded_calls == 1
    assert [item.github_id for item in iter_repositories(output)] == [1, 2]
    manifest = state.get_job(run_id, "discovery-manifest")
    assert manifest is not None
    assert manifest.payload["mode"] == "bounded"
    bounded_job = state.get_job(run_id, "bounded:00000000")
    assert bounded_job is not None
    assert bounded_job.status == "completed"
    assert [job.job_id for job in state.jobs(run_id, "repository-discovery-fetch")] == [
        "bounded:00000000"
    ]
    state.close()


@pytest.mark.asyncio
async def test_discovery_run_only_plans_lossless_partitions_without_a_limit(
    tmp_path: Path,
) -> None:
    run_id = "lossless-run"
    output = tmp_path / "repositories.jsonl"
    state = StateStore(tmp_path / "state.sqlite3")
    state.create_run(run_id, "discover", {})
    discovery = FakeLosslessDiscovery(run_id)

    with RepositoryJsonlAppender(output, fsync_interval=1) as writer:
        total, appended = await DiscoveryRun(
            discovery=cast(GitHubDiscovery, discovery),
            writer=writer,
            state=state,
            run_id=run_id,
            output=output,
            resume=False,
        ).run()

    assert total == appended == 1
    assert discovery.partition_calls == 1
    manifest = state.get_job(run_id, "discovery-manifest")
    assert manifest is not None
    assert manifest.payload["mode"] == "lossless"
    assert [job.job_id for job in state.jobs(run_id, "repository-discovery-fetch")] == [
        "partition:00000000"
    ]
    state.close()


@pytest.mark.asyncio
async def test_completed_discovery_replans_current_config_and_only_appends_unseen_outputs(
    tmp_path: Path,
) -> None:
    run_id = "expanded-bounded-run"
    output = tmp_path / "repositories.jsonl"
    state = StateStore(tmp_path / "state.sqlite3")
    state.create_run(run_id, "discover", {})

    first = FakeBoundedDiscovery(run_id, maximum=2)
    with RepositoryJsonlAppender(output, fsync_interval=1) as writer:
        total, appended = await DiscoveryRun(
            discovery=cast(GitHubDiscovery, first),
            writer=writer,
            state=state,
            run_id=run_id,
            output=output,
            resume=False,
        ).run()
    assert total == appended == 2

    state.finish_run(run_id)
    state.resume_run(run_id, "discover", {})
    expanded = FakeBoundedDiscovery(run_id, maximum=5)
    with RepositoryJsonlAppender(output, fsync_interval=1) as writer:
        total, appended = await DiscoveryRun(
            discovery=cast(GitHubDiscovery, expanded),
            writer=writer,
            state=state,
            run_id=run_id,
            output=output,
            resume=True,
        ).run()

    assert total == 5
    assert appended == 3
    assert [item.github_id for item in iter_repositories(output)] == [1, 2, 3, 4, 5]
    assert expanded.bounded_calls == 1
    state.close()


@pytest.mark.asyncio
async def test_discovery_resumes_the_same_run_after_durable_jsonl_checkpoint(
    tmp_path: Path,
) -> None:
    run_id = "resume-run"
    output = tmp_path / "repositories.jsonl"
    state = StateStore(tmp_path / "state.sqlite3")
    state.create_run(run_id, "discover", {})

    first = FakeTargetDiscovery(run_id, fail_on="owner/repo-2")
    with (
        RepositoryJsonlAppender(output, fsync_interval=1) as writer,
        pytest.raises(RuntimeError, match="simulated interruption"),
    ):
        await DiscoveryRun(
            discovery=cast(GitHubDiscovery, first),
            writer=writer,
            state=state,
            run_id=run_id,
            output=output,
            resume=False,
        ).run()
    assert [item.full_name for item in iter_repositories(output)] == ["owner/repo-1"]
    with output.open("a", encoding="utf-8") as handle:
        handle.write('{"discovery_run_id":"resume-run"')

    state.finish_run(run_id, error="simulated interruption")
    state.resume_run(run_id, "discover", {})
    resumed = FakeTargetDiscovery(run_id, fail_on=None)
    events = RecordingEventSink()
    with RepositoryJsonlAppender(output, fsync_interval=1) as writer:
        total, appended = await DiscoveryRun(
            discovery=cast(GitHubDiscovery, resumed),
            writer=writer,
            state=state,
            run_id=run_id,
            output=output,
            resume=True,
            events=events,
        ).run()

    assert total == 3
    assert appended == 2
    assert resumed.calls == ["owner/repo-2", "owner/repo-3"]
    assert [item.full_name for item in iter_repositories(output)] == [
        "owner/repo-1",
        "owner/repo-2",
        "owner/repo-3",
    ]
    progress = [event for event in events.events if event.kind == "discovery_progress"]
    assert [event.data["phase"] for event in progress] == [
        "fetching",
        "fetching",
        "fetching",
        "completed",
    ]
    assert progress[0].data["progress"] == pytest.approx(1 / 3)
    assert progress[-1].data["progress"] == 1.0
    state.close()


@pytest.mark.asyncio
async def test_discovery_run_uses_external_txt_targets_as_checkpointed_jobs(
    tmp_path: Path,
) -> None:
    run_id = "txt-target-run"
    output = tmp_path / "repositories.jsonl"
    state = StateStore(tmp_path / "state.sqlite3")
    state.create_run(run_id, "discover", {})
    discovery = FakeTargetDiscovery(run_id, fail_on=None, configured_targets=())

    with RepositoryJsonlAppender(output, fsync_interval=1) as writer:
        total, appended = await DiscoveryRun(
            discovery=cast(GitHubDiscovery, discovery),
            writer=writer,
            state=state,
            run_id=run_id,
            output=output,
            resume=False,
            target_repositories=("owner/repo-2", "owner/repo-3"),
        ).run()

    assert total == appended == 2
    assert discovery.calls == ["owner/repo-2", "owner/repo-3"]
    manifest = state.get_job(run_id, "discovery-manifest")
    assert manifest is not None
    assert manifest.payload["mode"] == "targets"
    assert manifest.payload["target_count"] == 2
    target_digest = manifest.payload["target_digest"]
    assert isinstance(target_digest, str) and len(target_digest) == 64
    jobs = state.jobs(run_id, "repository-discovery-fetch")
    assert [job.payload["full_name"] for job in jobs] == [
        "owner/repo-2",
        "owner/repo-3",
    ]
    state.close()


def test_repository_jsonl_rejects_duplicate_physical_identities(tmp_path: Path) -> None:
    output = tmp_path / "repositories.jsonl"
    item = candidate("run", 1)
    output.write_text(
        item.model_dump_json() + "\n" + item.model_dump_json() + "\n",
        encoding="utf-8",
    )
    with (
        pytest.raises(ValueError, match="duplicate repository identity"),
        RepositoryJsonlAppender(output),
    ):
        pass


def test_repository_contract_does_not_carry_unused_homepage() -> None:
    payload = candidate("run", 1).model_dump(mode="json")
    with pytest.raises(ValueError, match="homepage"):
        RepositoryCandidate.model_validate({**payload, "homepage": "https://example.com"})
