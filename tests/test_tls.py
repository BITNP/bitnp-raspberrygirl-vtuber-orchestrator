import ssl
from pathlib import Path

import pytest

from orchestrator.config import ConfigParseError
from orchestrator.tls import build_tls_context
from tests.private_ca import PrivateCA, create_private_ca


@pytest.fixture
def ca_path(tmp_path: Path) -> Path:
    certificate = ssl.create_default_context().get_ca_certs(binary_form=True)[0]
    path = tmp_path / "ca.pem"
    _ = path.write_text(ssl.DER_cert_to_PEM_cert(certificate), encoding="ascii")
    return path


def test_build_tls_context_returns_none_when_ca_path_is_absent() -> None:
    context = build_tls_context(None)

    assert context is None


def test_build_tls_context_uses_configured_pem_ca_bundle(ca_path: Path) -> None:
    context = build_tls_context(ca_path)

    assert isinstance(context, ssl.SSLContext)
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED


def test_build_tls_context_rejects_missing_ca_bundle(tmp_path: Path) -> None:
    with pytest.raises(ConfigParseError) as error:
        _ = build_tls_context(tmp_path / "missing-ca.pem")

    assert str(error.value) == "config field is blank: ORCHESTRATOR_TLS_CA_PATH"


def test_build_tls_context_rejects_unloadable_ca_bundle(tmp_path: Path) -> None:
    ca_path = tmp_path / "invalid-ca.pem"
    _ = ca_path.write_text("not a PEM certificate", encoding="ascii")

    with pytest.raises(ConfigParseError) as error:
        _ = build_tls_context(ca_path)

    assert str(error.value) == "config field is blank: ORCHESTRATOR_TLS_CA_PATH"


@pytest.fixture(scope="module")
def private_ca(tmp_path_factory: pytest.TempPathFactory) -> PrivateCA:
    return create_private_ca(tmp_path_factory.mktemp("tls-ca"))


def _handshake(client_context: ssl.SSLContext, ca: PrivateCA, hostname: str) -> None:
    """Exercise OpenSSL verification without sockets or background threads."""
    client_in, client_out = ssl.MemoryBIO(), ssl.MemoryBIO()
    server_in, server_out = ssl.MemoryBIO(), ssl.MemoryBIO()
    client = client_context.wrap_bio(client_in, client_out, server_hostname=hostname)
    server = ca.server_context().wrap_bio(server_in, server_out, server_side=True)
    client_done = server_done = False
    for _ in range(20):
        if not client_done:
            try:
                client.do_handshake()
                client_done = True
            except ssl.SSLWantReadError:
                pass
        if client_out.pending:
            _ = server_in.write(client_out.read())
        if not server_done:
            try:
                server.do_handshake()
                server_done = True
            except ssl.SSLWantReadError:
                pass
        if server_out.pending:
            _ = client_in.write(server_out.read())
        if client_done and server_done:
            return
    pytest.fail("内存 TLS 握手未完成")


def test_configured_private_ca_completes_real_tls_handshake(
    private_ca: PrivateCA,
) -> None:
    context = build_tls_context(private_ca.ca_path)
    assert context is not None
    _handshake(context, private_ca, "localhost")


@pytest.mark.parametrize("trust", ["default", "unrelated", "hostname_mismatch"])
def test_tls_rejects_untrusted_ca_or_hostname(
    private_ca: PrivateCA, trust: str
) -> None:
    path = private_ca.unrelated_ca_path if trust == "unrelated" else private_ca.ca_path
    context = (
        ssl.create_default_context() if trust == "default" else build_tls_context(path)
    )
    assert context is not None
    hostname = "127.0.0.1" if trust == "hostname_mismatch" else "localhost"
    with pytest.raises(ssl.SSLCertVerificationError):
        _handshake(context, private_ca, hostname)
