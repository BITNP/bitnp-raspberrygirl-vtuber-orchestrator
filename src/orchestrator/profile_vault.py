"""模块契约说明.

职责: 提供 orchestrator.profile_vault
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

import os
from base64 import b64encode
from hashlib import sha256
from pathlib import Path
from typing import final

from orchestrator.identity import EncryptedVoiceTemplate, VoiceProfileId


@final
class FileVoiceProfileVault:
    """类契约说明.

    职责: 定义 FileVoiceProfileVault
    的状态、行为和对外协作边界。
    契约: 方法: __init__、store_encrypted、del
    ete、_path。
    """

    def __init__(self, directory: Path, session_id: str = "") -> None:
        """函数契约说明.

        功能: 初始化 FileVoiceProfileVault
        的字段并建立实例不变式。
        参数: self 表示当前实例。 directory:
        Path。 必填。 session_id: str。 可省略。
        契约: 同步调用。 返回 `None`。
        """
        self._directory = directory

        self._session_id = session_id

        _ = directory.mkdir(mode=0o700, parents=True, exist_ok=True)

        _ = directory.chmod(0o700)

    def store_encrypted(
        self,
        profile_id: VoiceProfileId,
        template: EncryptedVoiceTemplate,
    ) -> None:
        """函数契约说明.

        功能: 执行 store_encrypted 的同步逻辑,并协调
        _path, with_suffix, chmod,
        replace。
        参数: self 表示当前实例。 profile_id:
        VoiceProfileId。 必填。 template:
        EncryptedVoiceTemplate。 必填。
        契约: 同步调用。 返回 `None`。
        """
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
        """函数契约说明.

        功能: 执行 delete 的同步逻辑,并协调 _path,
        exists, unlink,
        _fsync_directory。
        参数: self 表示当前实例。 profile_id:
        VoiceProfileId。 必填。
        契约: 同步调用。 返回 `None`。
        """
        path = self._path(profile_id)

        if path.exists():
            path.unlink()

            _fsync_directory(self._directory)

    def _path(self, profile_id: VoiceProfileId) -> Path:
        """函数契约说明.

        功能: 执行 _path 的同步逻辑,并协调
        hexdigest, sha256, encode。
        参数: self 表示当前实例。 profile_id:
        VoiceProfileId。 必填。
        契约: 同步调用。 返回 `Path`。
        """
        digest = sha256(f"{self._session_id}:{profile_id}".encode()).hexdigest()

        return self._directory / f"{digest}.template"


def _fsync_directory(directory: Path) -> None:
    """函数契约说明.

    功能: 执行 _fsync_directory 的同步逻辑,并协调
    open, fsync, close。
    参数: directory: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """
    descriptor = os.open(directory, os.O_RDONLY)

    try:
        os.fsync(descriptor)

    finally:
        os.close(descriptor)
