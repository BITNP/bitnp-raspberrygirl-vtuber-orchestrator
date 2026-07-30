
import os
from base64 import b64encode
from hashlib import sha256
from pathlib import Path
from typing import final

from orchestrator.identity import EncryptedVoiceTemplate, VoiceProfileId


@final
class FileVoiceProfileVault:

    def __init__(self, directory: Path, session_id: str = "") -> None:
        self._directory = directory

        self._session_id = session_id

        _ = directory.mkdir(mode=0o700, parents=True, exist_ok=True)

        _ = directory.chmod(0o700)

    def store_encrypted(
        self,
        profile_id: VoiceProfileId,
        template: EncryptedVoiceTemplate,
    ) -> None:
        path = self._path(profile_id)

        temporary = path.with_suffix(".tmp")

        with temporary.open("wb") as file:
            _ = file.write(b64encode(template.ciphertext))

            _ = file.flush()

            os.fsync(file.fileno())

        _ = temporary.chmod(0o600)

        _ = temporary.replace(path)

        _fsync_directory(self._directory)

    def delete(self, profile_id: VoiceProfileId) -> None:
        path = self._path(profile_id)

        if path.exists():
            path.unlink()

            _fsync_directory(self._directory)

    def _path(self, profile_id: VoiceProfileId) -> Path:
        digest = sha256(f"{self._session_id}:{profile_id}".encode()).hexdigest()

        return self._directory / f"{digest}.template"


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)

    try:
        os.fsync(descriptor)

    finally:
        os.close(descriptor)
