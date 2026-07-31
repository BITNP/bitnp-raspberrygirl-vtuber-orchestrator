from __future__ import annotations

import json
import ssl
import subprocess
import threading
from dataclasses import dataclass, field
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, ClassVar, Final, Self, cast, final, override

from websockets.sync.server import Server, ServerConnection, serve

if TYPE_CHECKING:
    import socket
    from pathlib import Path
    from types import TracebackType

TEST_HOSTNAME: Final = "localhost"
_OPENSSL_EXECUTABLE: Final = "/usr/bin/openssl"
_RUN_OPENSSL: Final = partial(
    subprocess.run,
    check=True,
    capture_output=True,
    text=True,
)


@dataclass(frozen=True, slots=True)
class PrivateCA:
    ca_path: Path
    unrelated_ca_path: Path
    certificate_path: Path
    key_path: Path

    def server_context(self) -> ssl.SSLContext:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self.certificate_path, self.key_path)
        return context


def create_private_ca(directory: Path) -> PrivateCA:
    ca_path = directory / "ca.pem"
    ca_key_path = directory / "ca-key.pem"
    certificate_path = directory / "server.pem"
    key_path = directory / "server-key.pem"
    certificate_request_path = directory / "server.csr"
    unrelated_ca_path = directory / "unrelated-ca.pem"
    unrelated_ca_key_path = directory / "unrelated-ca-key.pem"

    _create_ca(ca_path, ca_key_path, "orchestrator-test-ca")
    _create_ca(unrelated_ca_path, unrelated_ca_key_path, "unrelated-test-ca")
    _run_openssl(
        "req",
        "-new",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        str(key_path),
        "-out",
        str(certificate_request_path),
        "-subj",
        f"/CN={TEST_HOSTNAME}",
        "-addext",
        f"subjectAltName=DNS:{TEST_HOSTNAME}",
    )
    _run_openssl(
        "x509",
        "-req",
        "-in",
        str(certificate_request_path),
        "-CA",
        str(ca_path),
        "-CAkey",
        str(ca_key_path),
        "-CAcreateserial",
        "-out",
        str(certificate_path),
        "-days",
        "1",
        "-copy_extensions",
        "copy",
    )

    return PrivateCA(ca_path, unrelated_ca_path, certificate_path, key_path)


def _create_ca(certificate_path: Path, key_path: Path, common_name: str) -> None:
    _run_openssl(
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        str(key_path),
        "-out",
        str(certificate_path),
        "-days",
        "1",
        "-subj",
        f"/CN={common_name}",
        "-addext",
        "basicConstraints=critical,CA:TRUE",
        "-addext",
        "keyUsage=critical,keyCertSign,cRLSign",
    )


def _run_openssl(*arguments: str) -> None:
    _ = _RUN_OPENSSL((_OPENSSL_EXECUTABLE, *arguments))


@dataclass(slots=True)
class PrivateHttpsServer:
    private_ca: PrivateCA
    request_paths: list[str] = field(default_factory=list)
    _server: ThreadingHTTPServer = field(init=False)
    _thread: threading.Thread = field(init=False)

    def __post_init__(self) -> None:
        _PrivateHttpsHandler.request_paths = self.request_paths
        self._server = _PrivateHttpsServer(
            self.private_ca.server_context(),
            ("127.0.0.1", 0),
            _PrivateHttpsHandler,
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def endpoint(self) -> str:
        return f"https://{TEST_HOSTNAME}:{self._server.server_port}/v1"

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        _ = (exception_type, exception, traceback)
        self._server.shutdown()
        self._thread.join()
        self._server.server_close()


class _PrivateHttpsHandler(BaseHTTPRequestHandler):
    request_paths: ClassVar[list[str]]

    def do_POST(self) -> None:
        self.request_paths.append(self.path)
        _ = self.rfile.read(int(self.headers.get("content-length", "0")))
        body, content_type = _https_response(self.path)
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        _ = self.wfile.write(body)

    @override
    def log_message(self, format: str, *arguments: object) -> None:
        _ = (format, arguments)


@final
class _PrivateHttpsServer(ThreadingHTTPServer):
    daemon_threads: bool = True
    _context: ssl.SSLContext

    def __init__(
        self,
        context: ssl.SSLContext,
        server_address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
    ) -> None:
        self._context = context
        super().__init__(server_address, handler)

    @override
    def get_request(self) -> tuple[ssl.SSLSocket, tuple[str, int]]:
        request, address = cast(
            "tuple[socket.socket, tuple[str, int]]", self.socket.accept()
        )
        return (
            self._context.wrap_socket(
                request,
                server_side=True,
                do_handshake_on_connect=False,
            ),
            address,
        )


def _https_response(path: str) -> tuple[bytes, str]:
    if path.endswith("/chat/completions"):
        return (
            json.dumps({"choices": [{"message": {"content": "private CA"}}]}).encode(),
            "application/json",
        )
    if path.endswith("/audio/transcriptions"):
        return (
            json.dumps({"text": "private CA transcription"}).encode(),
            "application/json",
        )
    return b"private-ca-audio", "audio/wav"


@dataclass(slots=True)
class PrivateWssServer:
    private_ca: PrivateCA
    received_messages: list[str | bytes] = field(default_factory=list)
    _server: Server = field(init=False)
    _thread: threading.Thread = field(init=False)

    def __post_init__(self) -> None:
        self._server = serve(
            self._handle_connection,
            "127.0.0.1",
            0,
            ssl=self.private_ca.server_context(),
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def endpoint(self) -> str:
        address = cast("tuple[str, int]", self._server.socket.getsockname())
        assert isinstance(address, tuple)
        return f"wss://{TEST_HOSTNAME}:{address[1]}"

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        _ = (exception_type, exception, traceback)
        self._server.shutdown()
        self._thread.join()

    def _handle_connection(self, connection: ServerConnection) -> None:
        for _ in range(3):
            self.received_messages.append(connection.recv())
        connection.send('{"text":"private CA transcription","is_final":true}')
