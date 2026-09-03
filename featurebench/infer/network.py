"""Network isolation for agent execution.

Task containers keep normal network access during trusted setup and agent
installation. Immediately before the agent starts, all Docker networks are
detached. A container-local loopback bridge reaches a small allow-listing HTTP
CONNECT proxy through a read-only Unix socket mounted from the Docker host.
"""

from __future__ import annotations

import logging
import select
import socket
import socketserver
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from urllib.parse import urlsplit

import docker
from docker.models.containers import Container

from featurebench.infer.container import DOCKER_HOST_GATEWAY, ContainerManager


_MAX_HEADER_BYTES = 64 * 1024
_CONNECT_TIMEOUT_SECONDS = 15
_LOCAL_API_HOSTS = {
    DOCKER_HOST_GATEWAY,
    "host.docker.internal",
    "localhost",
    "127.0.0.1",
    "::1",
}


@dataclass(frozen=True, order=True)
class ApiEndpoint:
    """One host and TCP port that the agent may reach through the proxy."""

    host: str
    port: int

    def __post_init__(self) -> None:
        normalized_host = self.host.strip().rstrip(".").lower()
        if not normalized_host:
            raise ValueError("API endpoint host cannot be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError(f"Invalid API endpoint port: {self.port}")
        object.__setattr__(self, "host", normalized_host)


def _endpoint_from_url(value: str) -> ApiEndpoint:
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(
            f"Model API base URL must be an absolute HTTP(S) URL, got {value!r}"
        )
    return ApiEndpoint(
        parsed.hostname,
        parsed.port or (443 if parsed.scheme == "https" else 80),
    )


def _provider_from_model(model: str) -> str:
    raw = str(model or "").strip().lower()
    return raw.split("/", 1)[0] if "/" in raw else ""


def resolve_model_api_endpoints(
    agent: str,
    model: str,
    env_vars: Dict[str, str],
) -> Tuple[ApiEndpoint, ...]:
    """Resolve the model API origin used by one inference runner.

    Explicit base URLs always win.  Provider defaults cover the official APIs;
    an unknown provider must supply ``--base-url`` so the isolation boundary is
    unambiguous.
    """

    base_url_keys = {
        "openhands": ("LLM_BASE_URL",),
        "claude_code": ("ANTHROPIC_BASE_URL",),
        "gemini_cli": ("GOOGLE_GEMINI_BASE_URL",),
        "codex": ("OPENAI_BASE_URL",),
        "mini_swe_agent": (
            "MSWEA_BASE_URL",
            "ANTHROPIC_BASE_URL",
            "OPENAI_BASE_URL",
            "AZURE_API_BASE",
            "GOOGLE_GEMINI_BASE_URL",
        ),
    }
    explicit = []
    for key in base_url_keys.get(agent, ()):
        value = str(env_vars.get(key) or "").strip()
        if value:
            explicit.append(_endpoint_from_url(value))
    if explicit:
        endpoints = set(explicit)
    else:
        provider = _provider_from_model(model)
        if agent == "claude_code":
            provider = "anthropic"
        elif agent == "gemini_cli":
            provider = "google"
        elif agent == "codex":
            provider = "openai"

        provider_defaults = {
            "anthropic": ApiEndpoint("api.anthropic.com", 443),
            "openai": ApiEndpoint("api.openai.com", 443),
            "google": ApiEndpoint("generativelanguage.googleapis.com", 443),
            "gemini": ApiEndpoint("generativelanguage.googleapis.com", 443),
            "xai": ApiEndpoint("api.x.ai", 443),
            "openrouter": ApiEndpoint("openrouter.ai", 443),
            "together": ApiEndpoint("api.together.xyz", 443),
            "togetherai": ApiEndpoint("api.together.xyz", 443),
            "together_ai": ApiEndpoint("api.together.xyz", 443),
            "deepseek": ApiEndpoint("api.deepseek.com", 443),
            "mistral": ApiEndpoint("api.mistral.ai", 443),
            "groq": ApiEndpoint("api.groq.com", 443),
            "cohere": ApiEndpoint("api.cohere.com", 443),
        }
        endpoint = provider_defaults.get(provider)
        if endpoint is None:
            raise ValueError(
                f"Cannot determine the model API endpoint for agent={agent!r}, "
                f"model={model!r}. Pass --base-url explicitly."
            )
        endpoints = {endpoint}

    # Agent setup scripts translate loopback base URLs to the normal Docker
    # bridge gateway.  The restricted proxy sees that translated destination,
    # so allow the exact translated origin as well.
    for endpoint in tuple(endpoints):
        if endpoint.host in {"localhost", "127.0.0.1", "::1"}:
            endpoints.add(ApiEndpoint(DOCKER_HOST_GATEWAY, endpoint.port))

    return tuple(sorted(endpoints))


def _parse_authority(authority: str, default_port: int) -> ApiEndpoint:
    parsed = urlsplit(f"//{authority}")
    if not parsed.hostname:
        raise ValueError("Missing proxy request host")
    return ApiEndpoint(parsed.hostname, parsed.port or default_port)


def _read_headers(sock: socket.socket) -> bytes:
    data = bytearray()
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(min(8192, _MAX_HEADER_BYTES - len(data)))
        if not chunk:
            break
        data.extend(chunk)
        if len(data) >= _MAX_HEADER_BYTES:
            raise ValueError("Proxy request headers are too large")
    if b"\r\n\r\n" not in data:
        raise ValueError("Incomplete proxy request headers")
    return bytes(data)


def _relay(left: socket.socket, right: socket.socket) -> None:
    sockets = [left, right]
    for sock in sockets:
        sock.settimeout(None)
    while sockets:
        readable, _, exceptional = select.select(sockets, [], sockets)
        if exceptional:
            return
        for source in readable:
            target = right if source is left else left
            data = source.recv(64 * 1024)
            if not data:
                return
            target.sendall(data)


class _AllowlistProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: Tuple[str, int],
        allowed_endpoints: Iterable[ApiEndpoint],
        upstream_proxy: Optional[Tuple[str, int]],
        logger: logging.Logger,
    ):
        self.allowed_endpoints = frozenset(allowed_endpoints)
        self.upstream_proxy = upstream_proxy
        self.logger = logger
        super().__init__(server_address, _AllowlistProxyHandler)

    def handle_error(self, request, client_address) -> None:
        self.logger.debug("Restricted API proxy connection failed", exc_info=True)

    def connect_target(self, endpoint: ApiEndpoint) -> socket.socket:
        if self.upstream_proxy is None or endpoint.host in _LOCAL_API_HOSTS:
            host = "127.0.0.1" if endpoint.host in _LOCAL_API_HOSTS else endpoint.host
            return socket.create_connection(
                (host, endpoint.port), timeout=_CONNECT_TIMEOUT_SECONDS
            )

        upstream = socket.create_connection(
            self.upstream_proxy, timeout=_CONNECT_TIMEOUT_SECONDS
        )
        authority_host = (
            f"[{endpoint.host}]" if ":" in endpoint.host else endpoint.host
        )
        authority = f"{authority_host}:{endpoint.port}"
        upstream.sendall(
            f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n\r\n".encode("ascii")
        )
        response = _read_headers(upstream)
        status_line = response.split(b"\r\n", 1)[0]
        if not status_line.startswith(b"HTTP/") or b" 200 " not in status_line:
            upstream.close()
            raise ConnectionError(
                f"Upstream proxy rejected model API connection: "
                f"{status_line.decode('latin-1', errors='replace')}"
            )
        return upstream


class _AllowlistUnixProxyServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def __init__(
        self,
        socket_path: str,
        allowed_endpoints: Iterable[ApiEndpoint],
        upstream_proxy: Optional[Tuple[str, int]],
        logger: logging.Logger,
    ):
        self.allowed_endpoints = frozenset(allowed_endpoints)
        self.upstream_proxy = upstream_proxy
        self.logger = logger
        super().__init__(socket_path, _AllowlistProxyHandler)

    handle_error = _AllowlistProxyServer.handle_error
    connect_target = _AllowlistProxyServer.connect_target


class _AllowlistProxyHandler(socketserver.BaseRequestHandler):
    server: _AllowlistProxyServer

    def handle(self) -> None:
        self.request.settimeout(_CONNECT_TIMEOUT_SECONDS)
        try:
            request_data = _read_headers(self.request)
            header_data, body_prefix = request_data.split(b"\r\n\r\n", 1)
            lines = header_data.split(b"\r\n")
            method, target, version = lines[0].decode("latin-1").split(" ", 2)

            if method.upper() == "CONNECT":
                endpoint = _parse_authority(target, 443)
                outbound = self._open_allowed(endpoint)
                self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                if body_prefix:
                    outbound.sendall(body_prefix)
                _relay(self.request, outbound)
                return

            host_header = ""
            for line in lines[1:]:
                if line.lower().startswith(b"host:"):
                    host_header = line.split(b":", 1)[1].decode("latin-1").strip()
                    break
            parsed = urlsplit(target)
            if parsed.scheme:
                endpoint = ApiEndpoint(
                    parsed.hostname or "",
                    parsed.port or (443 if parsed.scheme == "https" else 80),
                )
                origin_target = parsed.path or "/"
                if parsed.query:
                    origin_target = f"{origin_target}?{parsed.query}"
            else:
                endpoint = _parse_authority(host_header, 80)
                origin_target = target

            outbound = self._open_allowed(endpoint)
            # ``connect_target`` establishes a tunnel even when an upstream
            # proxy is configured, so the destination always expects origin
            # form rather than proxy absolute form here.
            lines[0] = f"{method} {origin_target} {version}".encode("latin-1")
            outbound.sendall(b"\r\n".join(lines) + b"\r\n\r\n" + body_prefix)
            _relay(self.request, outbound)
        except PermissionError as exc:
            self.server.logger.info(str(exc))
            self.request.sendall(
                b"HTTP/1.1 403 Forbidden\r\nConnection: close\r\n"
                b"Content-Length: 0\r\n\r\n"
            )
        except Exception as exc:
            self.server.logger.debug("Restricted API proxy error: %s", exc)
            try:
                self.request.sendall(
                    b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n"
                    b"Content-Length: 0\r\n\r\n"
                )
            except OSError:
                pass
        finally:
            outbound_socket = locals().get("outbound")
            if outbound_socket is not None:
                outbound_socket.close()

    def _open_allowed(self, endpoint: ApiEndpoint) -> socket.socket:
        if endpoint not in self.server.allowed_endpoints:
            raise PermissionError(
                f"Blocked non-model network destination: {endpoint.host}:{endpoint.port}"
            )
        return self.server.connect_target(endpoint)


class AgentNetworkIsolation:
    """Own the internal Docker network and API-only proxy for one infer run."""

    def __init__(
        self,
        agent: str,
        model: str,
        env_vars: Dict[str, str],
        *,
        upstream_proxy_port: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.agent = agent
        self.model = model
        self.env_vars = dict(env_vars)
        self.upstream_proxy_port = upstream_proxy_port
        self.logger = logger or logging.getLogger(__name__)
        self.client = docker.from_env()
        self.endpoints = resolve_model_api_endpoints(agent, model, self.env_vars)
        self.proxy: Optional[_AllowlistUnixProxyServer] = None
        self.proxy_thread: Optional[threading.Thread] = None
        self.socket_dir: Optional[tempfile.TemporaryDirectory[str]] = None
        self.socket_path: Optional[Path] = None

    def start(self) -> None:
        if self.proxy is not None:
            return
        socket_dir = tempfile.TemporaryDirectory(
            prefix=f"fb-api-only-{uuid.uuid4().hex[:8]}-"
        )
        socket_path = Path(socket_dir.name) / "proxy.sock"
        proxy = None
        try:
            upstream = None
            if self.upstream_proxy_port is not None:
                upstream = ("127.0.0.1", self.upstream_proxy_port)
            proxy = _AllowlistUnixProxyServer(
                str(socket_path), self.endpoints, upstream, self.logger
            )
            socket_path.chmod(0o777)
            thread = threading.Thread(
                target=proxy.serve_forever,
                name="featurebench-agent-api-only-proxy",
                daemon=True,
            )
            thread.start()
        except BaseException:
            if proxy is not None:
                proxy.server_close()
            socket_dir.cleanup()
            raise

        self.proxy = proxy
        self.proxy_thread = thread
        self.socket_dir = socket_dir
        self.socket_path = socket_path
        allowed = ", ".join(f"{item.host}:{item.port}" for item in self.endpoints)
        self.logger.info("Agent API-only network allows: %s", allowed)

    def docker_volume(self) -> Dict[str, Dict[str, str]]:
        """Return the read-only Unix-socket mount required by task containers."""

        if self.socket_dir is None:
            raise RuntimeError("Agent network isolation has not been started")
        return {
            self.socket_dir.name: {
                "bind": "/run/featurebench-network",
                "mode": "ro",
            }
        }

    def isolate(
        self,
        container: Container,
        container_manager: ContainerManager,
        log_file: Path,
    ) -> bool:
        """Move a prepared task container onto the API-only network."""

        if self.proxy is None or self.socket_path is None:
            raise RuntimeError("Agent network isolation has not been started")
        try:
            bridge_source = Path(__file__).with_name("proxy_bridge.py")
            container_manager.copy_to_container(
                container, bridge_source, "/installed-agent/fb_proxy_bridge.py"
            )
            command = (
                "mkdir -p /agent-logs; "
                "rm -f /installed-agent/fb-proxy-port; "
                "python_bin=\"$(command -v python3 || command -v python)\"; "
                "[ -n \"$python_bin\" ] || exit 127; "
                "nohup \"$python_bin\" /installed-agent/fb_proxy_bridge.py "
                "--socket /run/featurebench-network/proxy.sock "
                "--port-file /installed-agent/fb-proxy-port "
                ">/agent-logs/fb_proxy_bridge.log 2>&1 & "
                "for attempt in $(seq 1 100); do "
                "[ -s /installed-agent/fb-proxy-port ] && break; "
                "sleep 0.05; "
                "done; "
                "[ -s /installed-agent/fb-proxy-port ] || exit 1; "
                "proxy_port=\"$(cat /installed-agent/fb-proxy-port)\"; "
                "cat >> /installed-agent/setup-env.sh <<FB_NETWORK_EOF\n"
                "\n# FeatureBench agent network: model API only\n"
                "export HTTP_PROXY='http://127.0.0.1:$proxy_port'\n"
                "export HTTPS_PROXY='http://127.0.0.1:$proxy_port'\n"
                "export http_proxy='http://127.0.0.1:$proxy_port'\n"
                "export https_proxy='http://127.0.0.1:$proxy_port'\n"
                "export NO_PROXY='localhost,127.0.0.1,::1'\n"
                "export no_proxy='localhost,127.0.0.1,::1'\n"
                "unset ALL_PROXY all_proxy\n"
                "FB_NETWORK_EOF"
            )
            exit_code, output = container_manager.exec_command(
                container, command, log_file=log_file
            )
            if exit_code != 0:
                raise RuntimeError(f"Failed to configure restricted API proxy: {output}")

            container.reload()
            attached_networks = tuple(
                container.attrs.get("NetworkSettings", {}).get("Networks", {}).keys()
            )
            for network_name in attached_networks:
                self.client.networks.get(network_name).disconnect(container, force=True)

            container.reload()
            remaining = set(
                container.attrs.get("NetworkSettings", {}).get("Networks", {}).keys()
            )
            if remaining:
                raise RuntimeError(
                    f"Unexpected task-container networks after isolation: {sorted(remaining)}"
                )
            self.logger.info("Restricted agent network enabled for %s", container.short_id)
            return True
        except Exception as exc:
            self.logger.error("Failed to isolate agent network: %s", exc)
            return False

    def close(self) -> None:
        if self.proxy is not None:
            self.proxy.shutdown()
            self.proxy.server_close()
        if self.proxy_thread is not None:
            self.proxy_thread.join(timeout=5)
        if self.socket_dir is not None:
            self.socket_dir.cleanup()
        self.proxy = None
        self.proxy_thread = None
        self.socket_dir = None
        self.socket_path = None
