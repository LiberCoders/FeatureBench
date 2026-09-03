import json
import logging

import pytest

from featurebench.infer import run_infer
from featurebench.infer.agents.mini_swe_agent import MiniSweAgent
from featurebench.infer.models import InferResult


MISSING_TRAJECTORY = object()
MALFORMED_TRAJECTORY = object()


class FakeContainerManager:
    def __init__(self, exit_status, *, stream_exit_code=0):
        self.exit_status = exit_status
        self.stream_exit_code = stream_exit_code
        self.commands = []
        self.copied_to_container = []

    def exec_command(self, container, command, log_file=None):
        self.commands.append(command)
        return 0, ""

    def exec_command_stream(self, container, command, log_file=None, timeout=None):
        # mini-swe-agent returns process code 0 for semantic terminal statuses
        # such as LimitsExceeded.
        if self.stream_exit_code == -1 and log_file is not None:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n[TIMEOUT after {timeout} seconds]\n")
        return self.stream_exit_code

    def copy_from_container(self, container, source, destination):
        if source.endswith(".traj.json"):
            if self.exit_status is MISSING_TRAJECTORY:
                raise FileNotFoundError(source)
            if self.exit_status is MALFORMED_TRAJECTORY:
                destination.write_text("{not-json", encoding="utf-8")
                return
            destination.write_text(
                json.dumps({"info": {"exit_status": self.exit_status}}),
                encoding="utf-8",
            )
        else:
            destination.write_text("agent output", encoding="utf-8")

    def copy_to_container(self, container, source, destination):
        self.copied_to_container.append((source, destination))


def _agent(exit_status, *, cost_limit=None, stream_exit_code=0):
    return MiniSweAgent(
        FakeContainerManager(exit_status, stream_exit_code=stream_exit_code),
        env_vars={"MSWEA_API_KEY": "test-key"},
        logger=logging.getLogger("test-mini-swe-agent"),
        model="openai/test-model",
        cost_limit=cost_limit,
    )


def test_submitted_is_the_only_successful_terminal_status(tmp_path):
    agent = _agent("Submitted")

    assert agent.run(object(), "solve task", tmp_path / "infer.log")
    assert agent.agent_exit_status == "Submitted"


def test_run_uses_restricted_command_environment():
    command = _agent("Submitted").get_run_command("solve task")

    assert "featurebench_mini_swe_environment.RestrictedLocalEnvironment" in command
    assert "sys.path.insert(0, '/opt/featurebench-controller')" in command


def test_run_forwards_zero_cost_limit():
    command = _agent("Submitted", cost_limit=0.0).get_run_command("solve task")

    assert "--cost-limit 0.0 " in command


def test_run_uses_upstream_cost_limit_when_unset():
    command = _agent("Submitted").get_run_command("solve task")

    assert "--cost-limit" not in command


def test_cli_parses_cost_limit_after_task_ids(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "fb-infer",
            "--agent",
            "mini_swe_agent",
            "--model",
            "openai/test-model",
            "--task-id",
            "task-one",
            "task-two",
            "--cost-limit",
            "0",
        ],
    )

    args = run_infer.parse_args()

    assert args.task_id == ["task-one", "task-two"]
    assert args.cost_limit == 0


def test_pre_run_setup_hides_controller_and_credentials(tmp_path):
    agent = _agent("Submitted")

    assert agent.pre_run_setup(object(), object(), tmp_path / "infer.log")
    assert agent.cm.copied_to_container[0][1] == (
        "/opt/featurebench-controller/featurebench_mini_swe_environment.py"
    )
    setup = "\n".join(agent.cm.commands)
    assert "chown -R fbagent:fbagent /testbed" in setup
    assert "chmod 700 /opt/mini-swe-agent-venv" in setup
    assert "chmod 700 /opt/featurebench-controller /installed-agent /agent-logs" in setup
    assert "test ! -r /opt/mini-swe-agent-venv/pyvenv.cfg" in setup
    assert "test ! -r /installed-agent/setup-env.sh" in setup


def test_inference_restores_mini_swe_workspace_ownership(tmp_path):
    manager = FakeContainerManager("Submitted")

    run_infer._restore_mini_swe_workspace_ownership(
        "mini_swe_agent",
        manager,
        object(),
        tmp_path / "infer.log",
    )

    assert manager.commands == ["chown -R root:root /testbed"]


def test_inference_does_not_change_other_agent_workspace_ownership(tmp_path):
    manager = FakeContainerManager("Submitted")

    run_infer._restore_mini_swe_workspace_ownership(
        "openhands",
        manager,
        object(),
        tmp_path / "infer.log",
    )

    assert manager.commands == []


@pytest.mark.parametrize(
    "exit_status",
    [
        "LimitsExceeded",
        "TimeExceeded",
        "RepeatedFormatError",
        "AuthenticationError",
        "FutureMiniSweAgentError",
    ],
)
def test_non_submission_status_fails_even_with_zero_process_exit(
    tmp_path, exit_status
):
    agent = _agent(exit_status)

    assert not agent.run(object(), "solve task", tmp_path / "infer.log")
    assert agent.agent_exit_status == exit_status


def test_missing_exit_status_is_failure(tmp_path):
    agent = _agent(None)

    assert not agent.run(object(), "solve task", tmp_path / "infer.log")
    assert agent.agent_exit_status is None


def test_outer_timeout_sets_time_exceeded_status(tmp_path):
    agent = _agent("Submitted", stream_exit_code=-1)

    assert not agent.run(
        object(),
        "solve task",
        tmp_path / "infer.log",
        timeout=3600,
    )
    assert agent.agent_exit_status == "TimeExceeded"


@pytest.mark.parametrize(
    "trajectory", [MISSING_TRAJECTORY, MALFORMED_TRAJECTORY]
)
def test_missing_or_malformed_trajectory_is_failure(tmp_path, trajectory):
    agent = _agent(trajectory)

    assert not agent.run(object(), "solve task", tmp_path / "infer.log")
    assert agent.agent_exit_status is None


def test_infer_result_serializes_agent_exit_status():
    result = InferResult(
        instance_id="owner__repo.commit.task.lv1",
        model_patch="",
        agent="mini_swe_agent",
        model="openai/test-model",
        n_attempt=1,
        metadata={},
        success=False,
        error="Agent exited with status: LimitsExceeded",
        agent_exit_status="LimitsExceeded",
    )

    assert result.to_dict()["agent_exit_status"] == "LimitsExceeded"
