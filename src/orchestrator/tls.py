import ssl
from pathlib import Path

from orchestrator.config import TLS_CA_PATH_KEY, ConfigParseError


def build_tls_context(ca_path: Path | None) -> ssl.SSLContext | None:
    if ca_path is None:
        return None

    context = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)

    try:
        context.load_verify_locations(cafile=str(ca_path))
    except (OSError, ssl.SSLError) as error:
        raise ConfigParseError(field_name=TLS_CA_PATH_KEY) from error

    return context
