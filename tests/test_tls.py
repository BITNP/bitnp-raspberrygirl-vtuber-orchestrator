import ssl
from pathlib import Path

import pytest

from orchestrator.config import ConfigParseError
from orchestrator.tls import build_tls_context


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
