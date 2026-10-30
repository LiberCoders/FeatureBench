import urllib.error
import urllib.request

from featurebench.infer.agents import openhands


def test_session_release_url_strips_v1_and_quotes_session_id():
    url = openhands._session_release_url(
        "http://10.43.2.226:7486/v1/",
        "task with/slash",
    )

    assert url == "http://10.43.2.226:7486/session_release?session_id=task+with%2Fslash"


def test_release_backend_session_sends_delete(monkeypatch):
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return False

        def read(self):
            return b""

    def fake_urlopen(req, timeout):
        requests.append((req, timeout))
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert openhands._release_backend_session(
        "http://backend.example/v1",
        "session-1",
        logger=None,
    )
    assert len(requests) == 1
    req, timeout = requests[0]
    assert req.get_method() == "DELETE"
    assert req.full_url == "http://backend.example/session_release?session_id=session-1"
    assert timeout == 10


def test_release_backend_session_returns_false_on_error(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert not openhands._release_backend_session(
        "http://backend.example/v1",
        "session-1",
        logger=None,
    )


def test_release_session_from_container_env_marks_and_skips_duplicate(monkeypatch):
    calls = []

    def fake_release(api_base, session_id, logger):
        calls.append((api_base, session_id))
        return True

    monkeypatch.setattr(openhands, "_release_backend_session", fake_release)

    class FakeContainer:
        def __init__(self):
            self.released = False

        def exec_run(self, cmd):
            script = cmd[-1]
            if script.startswith("test -f "):
                return (0 if self.released else 1, b"")
            if "LLM_SESSION_ID" in script:
                return (0, b"fb-session-1")
            if "LLM_BASE_URL" in script:
                return (0, b"http://backend.example/v1")
            if "session_release.done" in script:
                self.released = True
                return (0, b"")
            return (0, b"")

    agent = openhands.OpenHandsAgent(
        container_manager=None,
        env_vars={"LLM_BASE_URL": "http://fallback.example/v1"},
        logger=None,
    )
    container = FakeContainer()

    agent._release_session_from_container_env(container)
    agent._release_session_from_container_env(container)

    assert calls == [("http://backend.example/v1", "fb-session-1")]
    assert container.released
