import http.server
import logging
import socket
import threading

import pytest

from featurebench.infer.container import DOCKER_HOST_GATEWAY
from featurebench.infer.network import (
    ApiEndpoint,
    _AllowlistProxyServer,
    resolve_model_api_endpoints,
)


class _QuietHttpHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"allowed"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return None


def _start_server(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def _proxy_get(proxy_port, target_port):
    with socket.create_connection(("127.0.0.1", proxy_port), timeout=3) as client:
        client.sendall(
            f"GET http://127.0.0.1:{target_port}/ HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{target_port}\r\nConnection: close\r\n\r\n".encode()
        )
        chunks = []
        while True:
            chunk = client.recv(8192)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)


def test_resolve_model_api_endpoint_from_explicit_base_url():
    endpoints = resolve_model_api_endpoints(
        "mini_swe_agent",
        "openai/model",
        {"MSWEA_BASE_URL": "https://gateway.example/v1"},
    )

    assert endpoints == (ApiEndpoint("gateway.example", 443),)


def test_resolve_loopback_base_url_includes_translated_docker_gateway():
    endpoints = resolve_model_api_endpoints(
        "codex",
        "openai/model",
        {"OPENAI_BASE_URL": "http://localhost:8080/v1"},
    )

    assert ApiEndpoint("localhost", 8080) in endpoints
    assert ApiEndpoint(DOCKER_HOST_GATEWAY, 8080) in endpoints


def test_resolve_known_provider_default_and_reject_unknown_provider():
    assert resolve_model_api_endpoints(
        "mini_swe_agent", "deepseek/deepseek-v4", {}
    ) == (ApiEndpoint("api.deepseek.com", 443),)

    with pytest.raises(ValueError, match="Pass --base-url explicitly"):
        resolve_model_api_endpoints("mini_swe_agent", "custom/model", {})


def test_allowlist_proxy_allows_only_configured_origin():
    allowed_server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), _QuietHttpHandler
    )
    denied_server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), _QuietHttpHandler
    )
    allowed_thread = _start_server(allowed_server)
    denied_thread = _start_server(denied_server)

    allowed_port = allowed_server.server_address[1]
    denied_port = denied_server.server_address[1]
    proxy = _AllowlistProxyServer(
        ("127.0.0.1", 0),
        [ApiEndpoint("127.0.0.1", allowed_port)],
        None,
        logging.getLogger("test"),
    )
    proxy_thread = _start_server(proxy)
    try:
        allowed_response = _proxy_get(proxy.server_address[1], allowed_port)
        denied_response = _proxy_get(proxy.server_address[1], denied_port)
    finally:
        proxy.shutdown()
        proxy.server_close()
        allowed_server.shutdown()
        allowed_server.server_close()
        denied_server.shutdown()
        denied_server.server_close()
        proxy_thread.join(timeout=3)
        allowed_thread.join(timeout=3)
        denied_thread.join(timeout=3)

    assert b" 200 " in allowed_response.split(b"\r\n", 1)[0]
    assert allowed_response.endswith(b"allowed")
    assert b" 403 " in denied_response.split(b"\r\n", 1)[0]


def test_allowlist_proxy_can_chain_through_configured_upstream_proxy():
    target_server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), _QuietHttpHandler
    )
    target_thread = _start_server(target_server)
    target = ApiEndpoint("127.0.0.1", target_server.server_address[1])
    upstream = _AllowlistProxyServer(
        ("127.0.0.1", 0), [target], None, logging.getLogger("test")
    )
    upstream_thread = _start_server(upstream)
    downstream = _AllowlistProxyServer(
        ("127.0.0.1", 0),
        [target],
        ("127.0.0.1", upstream.server_address[1]),
        logging.getLogger("test"),
    )
    downstream_thread = _start_server(downstream)
    try:
        response = _proxy_get(
            downstream.server_address[1], target_server.server_address[1]
        )
    finally:
        downstream.shutdown()
        downstream.server_close()
        upstream.shutdown()
        upstream.server_close()
        target_server.shutdown()
        target_server.server_close()
        downstream_thread.join(timeout=3)
        upstream_thread.join(timeout=3)
        target_thread.join(timeout=3)

    assert b" 200 " in response.split(b"\r\n", 1)[0]
    assert response.endswith(b"allowed")
