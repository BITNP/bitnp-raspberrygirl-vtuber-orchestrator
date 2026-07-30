"""模块契约说明.

职责: 提供 orchestrator.release_certificate
模块的领域模型、边界函数和运行时协作逻辑。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from .json_boundary import JsonValue, parse_json_value

if TYPE_CHECKING:
    from pathlib import Path


FORMAT_VERSION: Final = "task-8-certificate-v1"

PLAN: Final = "core-loop-before-frontend"

TASK: Final = 8

MANIFEST_COUNT: Final = 2

CERTIFICATE_KEY_ENV: Final = "TASK8_CERTIFICATE_KEY"

EXPECTED_COMMANDS: Final = (
    ("bitnp-raspberrygirl-vtuber-orchestrator", "uv\u001fsync\u001f--locked"),
    ("bitnp-raspberrygirl-vtuber-orchestrator", "uv\u001frun\u001fpytest"),
    ("bitnp-raspberrygirl-vtuber-orchestrator", "uv\u001frun\u001fbasedpyright"),
    (
        "bitnp-raspberrygirl-vtuber-orchestrator",
        "uv\u001frun\u001fruff\u001fcheck\u001fsrc\u001ftests",
    ),
    (
        "bitnp-raspberrygirl-vtuber-orchestrator",
        "python\u001fscripts/verify_protocol_schema.py",
    ),
    (
        "bitnp-raspberrygirl-vtuber-orchestrator",
        "python\u001fscripts/verify_topology.py\u001f--sibling-root\u001f..",
    ),
    (
        "bitnp-raspberrygirl-vtuber-orchestrator",
        "python\u001fscripts/verify_vtuber_contract.py\u001f--frontend-path\u001f../bitnp-raspberrygirl-vtuber-frontend",
    ),
    (
        "bitnp-raspberrygirl-vtuber-orchestrator",
        "bash\u001fscripts/verify_workspace.sh\u001f--sibling-root\u001f..",
    ),
    ("bitnp-raspberrygirl-vtuber-mic", "uv\u001fsync\u001f--locked"),
    ("bitnp-raspberrygirl-vtuber-mic", "uv\u001frun\u001fpytest"),
    ("bitnp-raspberrygirl-vtuber-sound", "uv\u001fsync\u001f--locked"),
    ("bitnp-raspberrygirl-vtuber-sound", "uv\u001frun\u001fpytest"),
    ("bitnp-raspberrygirl-vtuber-comments", "uv\u001fsync\u001f--locked"),
    ("bitnp-raspberrygirl-vtuber-comments", "uv\u001frun\u001fpytest"),
    (
        "bitnp-raspberrygirl-vtuber-frontend",
        "godot\u001f--headless\u001f--path\u001f.\u001f--script\u001fres://tests/protocol_smoke.gd",
    ),
)


@dataclass(frozen=True, slots=True)
class CertificateRequest:
    """类契约说明.

    职责: 保存 CertificateRequest
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: certificate、first_manifest、s
    econd_manifest、frontend_baseline、pla
    n_digest、source_digest。
    """

    certificate: Path

    first_manifest: Path

    second_manifest: Path

    frontend_baseline: str

    plan_digest: str

    source_digest: str = ""


@dataclass(frozen=True, slots=True)
class CertificateVerification:
    """类契约说明.

    职责: 保存 CertificateVerification
    不可变数据结构,用类型标注表达字段契约。
    契约: 字段: accepted、code。
    """

    accepted: bool

    code: str


def write_certificate(request: CertificateRequest) -> None:
    """函数契约说明.

    功能: 执行 write_certificate 的同步逻辑,并协调
    _sha256, _unsigned_payload,
    _authority_key, write_text。
    参数: request: CertificateRequest。 必填。
    契约: 同步调用。 返回 `None`。 可能抛出
    RuntimeError。
    """
    first_digest = _sha256(request.first_manifest)

    second_digest = _sha256(request.second_manifest)

    if request.source_digest and (
        not _manifest_valid(request.first_manifest)
        or not _manifest_valid(request.second_manifest)
    ):
        message = "invalid_release_manifest"

        raise RuntimeError(message)

    unsigned = _unsigned_payload(request, first_digest, second_digest)

    key = _authority_key()

    if key is None:
        message = "certificate_authority_unavailable"

        raise RuntimeError(message)

    payload = {**unsigned, "signature": _sign_payload(unsigned, key)}

    _ = request.certificate.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def verify_certificate(
    request: CertificateRequest,
) -> CertificateVerification:
    """函数契约说明.

    功能: 校验相关输入、协议或运行时约束。
    参数: request: CertificateRequest。 必填。
    契约: 同步调用。 返回
    `CertificateVerification`。
    """
    try:
        raw = parse_json_value(request.certificate.read_text(encoding="utf-8"))

    except (OSError, json.JSONDecodeError):
        return CertificateVerification(accepted=False, code="invalid_certificate")

    if not isinstance(raw, dict):
        return CertificateVerification(accepted=False, code="invalid_certificate")

    key = _authority_key()

    if key is None:
        return CertificateVerification(
            accepted=False, code="certificate_authority_unavailable"
        )

    signature = raw.get("signature")

    if not isinstance(signature, str):
        return CertificateVerification(accepted=False, code="invalid_certificate")

    unsigned = {key: value for key, value in raw.items() if key != "signature"}

    if not hmac.compare_digest(signature, _sign_payload(unsigned, key)):
        return CertificateVerification(accepted=False, code="invalid_certificate")

    if not _identity_matches(unsigned, request):
        return CertificateVerification(accepted=False, code="invalid_certificate")

    manifest_hashes = unsigned.get("manifest_sha256")

    try:
        matches = _manifest_hashes_match(manifest_hashes, request)

    except OSError:
        return CertificateVerification(accepted=False, code="manifest_digest_mismatch")

    if not matches:
        return CertificateVerification(accepted=False, code="manifest_digest_mismatch")

    if request.source_digest and (
        not _manifest_valid(request.first_manifest)
        or not _manifest_valid(request.second_manifest)
    ):
        return CertificateVerification(accepted=False, code="invalid_release_manifest")

    return CertificateVerification(accepted=True, code="accepted")


def _unsigned_payload(
    request: CertificateRequest, first_digest: str, second_digest: str
) -> dict[str, JsonValue]:
    """函数契约说明.

    功能: 执行 _unsigned_payload 的同步逻辑,并协调
    str, resolve。
    参数: request: CertificateRequest。 必填。
    first_digest: str。 必填。
    second_digest: str。 必填。
    契约: 同步调用。 返回 `dict[str, JsonValue]`。
    """
    return {
        "format_version": FORMAT_VERSION,
        "plan": PLAN,
        "task": TASK,
        "plan_sha256": request.plan_digest,
        "frontend_baseline": request.frontend_baseline,
        "source_sha256": request.source_digest,
        "manifest_paths": [
            str(request.first_manifest.resolve()),
            str(request.second_manifest.resolve()),
        ],
        "manifest_sha256": [first_digest, second_digest],
        "passed": True,
    }


def _sign_payload(payload: dict[str, JsonValue], key: bytes) -> str:
    """函数契约说明.

    功能: 执行 _sign_payload 的同步逻辑,并协调
    encode, hexdigest, dumps, new。
    参数: payload: dict[str, JsonValue]。
    必填。 key: bytes。 必填。
    契约: 同步调用。 返回 `str`。
    """
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()

    return hmac.new(key, encoded, hashlib.sha256).hexdigest()


def _identity_matches(
    payload: dict[str, JsonValue], request: CertificateRequest
) -> bool:
    """函数契约说明.

    功能: 执行 _identity_matches 的同步逻辑,并协调
    get。
    参数: payload: dict[str, JsonValue]。
    必填。 request: CertificateRequest。 必填。
    契约: 同步调用。 返回 `bool`。
    """
    return (
        payload.get("format_version") == FORMAT_VERSION
        and payload.get("plan") == PLAN
        and payload.get("task") == TASK
        and payload.get("plan_sha256") == request.plan_digest
        and payload.get("frontend_baseline") == request.frontend_baseline
        and payload.get("source_sha256") == request.source_digest
        and payload.get("passed") is True
    )


def _manifest_hashes_match(
    value: JsonValue | None, request: CertificateRequest
) -> bool:
    """函数契约说明.

    功能: 执行 _manifest_hashes_match
    的同步逻辑,并协调 all, isinstance, len,
    _sha256。
    参数: value: JsonValue | None。 必填。
    request: CertificateRequest。 必填。
    契约: 同步调用。 返回 `bool`。
    """
    if not isinstance(value, list) or len(value) != MANIFEST_COUNT:
        return False

    if not all(isinstance(digest, str) for digest in value):
        return False

    return value[0] == _sha256(request.first_manifest) and value[1] == _sha256(
        request.second_manifest
    )


def _sha256(path: Path) -> str:
    """函数契约说明.

    功能: 执行 _sha256 的同步逻辑,并协调 hexdigest,
    sha256, read_bytes。
    参数: path: Path。 必填。
    契约: 同步调用。 返回 `str`。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authority_key() -> bytes | None:
    """函数契约说明.

    功能: 执行 _authority_key 的同步逻辑,并协调 get,
    encode。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `bytes | None`。
    """
    key = os.environ.get(CERTIFICATE_KEY_ENV)

    if key is None or key == "":
        return None

    return key.encode()


def _manifest_valid(path: Path) -> bool:
    """函数契约说明.

    功能: 执行 _manifest_valid 的同步逻辑,并协调
    get, enumerate, parse_json_value,
    read_text。
    参数: path: Path。 必填。
    契约: 同步调用。 返回 `bool`。
    """
    try:
        raw = parse_json_value(path.read_text(encoding="utf-8"))

    except (OSError, json.JSONDecodeError):
        return False

    if not isinstance(raw, dict) or raw.get("passed") is not True:
        return False

    records = raw.get("records")

    if not isinstance(records, list) or len(records) != len(EXPECTED_COMMANDS):
        return False

    for index, expected in enumerate(EXPECTED_COMMANDS):
        record = records[index]

        if not isinstance(record, dict):
            return False

        stdout = f"{index:02d}-{expected[0]}.stdout.log"

        stderr = f"{index:02d}-{expected[0]}.stderr.log"

        if (
            record.get("repository"),
            record.get("arguments"),
            record.get("returncode"),
        ) != (*expected, 0):
            return False

        if record.get("stdout_path") != stdout or record.get("stderr_path") != stderr:
            return False

        if record.get("stdout_raw_sha256") != _sha256(path.parent / stdout):
            return False

        if record.get("stderr_raw_sha256") != _sha256(path.parent / stderr):
            return False

    return True
