"""Bridge a container-local TCP proxy port to a host-mounted Unix socket."""

from __future__ import annotations

import argparse
import select
import socket
import socketserver
from pathlib import Path


def _relay(left: socket.socket, right: socket.socket) -> None:
    sockets = [left, right]
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


class _BridgeServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        super().__init__(("127.0.0.1", 0), _BridgeHandler)


class _BridgeHandler(socketserver.BaseRequestHandler):
    server: _BridgeServer

    def handle(self) -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
            upstream.connect(self.server.socket_path)
            _relay(self.request, upstream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--port-file", required=True)
    args = parser.parse_args()

    server = _BridgeServer(args.socket)
    port_file = Path(args.port_file)
    port_file.write_text(str(server.server_address[1]), encoding="utf-8")
    server.serve_forever()


if __name__ == "__main__":
    main()
