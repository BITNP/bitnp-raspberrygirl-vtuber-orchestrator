"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

import re
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[2]

SERVICE_SRC_DIRS: Final = (
    ROOT / "mic" / "src",
    ROOT / "asr" / "src",
    ROOT / "comments" / "src",
    ROOT / "tts" / "src",
    ROOT / "sound" / "src",
)

SCHEMA_DIR: Final = ROOT / "schemas" / "protocol"

SERVICE_MODE_PATTERN: Final = re.compile(
    r"\b(mode|lecturer|virtual_streamer|onsite_explainer|danmaku|ppt|script)\b",
    re.IGNORECASE,
)

SCHEMA_MODE_PATTERN: Final = re.compile(
    r"\b(lecturer|virtual_streamer|onsite_explainer|danmaku|ppt|script)\b",
    re.IGNORECASE,
)

MODE_AGNOSTIC_EVENT_PREFIXES: Final = (
    '"audience.input"',
    '"asr.',
    '"tts.',
    '"sound.',
)


def service_source_violations() -> tuple[str, ...]:
    """函数契约说明.

    功能: 执行 service_source_violations
    的同步逻辑,并协调 tuple, sorted, rglob,
    read_text。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `tuple[str, ...]`。
    """

    violations: list[str] = []

    for src_dir in SERVICE_SRC_DIRS:
        for path in sorted(src_dir.rglob("*.py")):
            text = path.read_text(encoding="utf-8")

            if SERVICE_MODE_PATTERN.search(text):
                violations.append(path.relative_to(ROOT).as_posix())

    return tuple(violations)


def schema_mode_violations() -> tuple[str, ...]:
    """函数契约说明.

    功能: 执行 schema_mode_violations
    的同步逻辑,并协调 sorted, tuple, rglob,
    splitlines。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `tuple[str, ...]`。
    """

    violations: list[str] = []

    for path in sorted(SCHEMA_DIR.rglob("*.json")):
        lines = path.read_text(encoding="utf-8").splitlines()

        for line_number, line in enumerate(lines, start=1):
            if SCHEMA_MODE_PATTERN.search(line):
                violations.append(f"{path.relative_to(ROOT).as_posix()}:{line_number}")

            line_is_mode_agnostic_event = line.strip().startswith(
                MODE_AGNOSTIC_EVENT_PREFIXES,
            )

            if line_is_mode_agnostic_event and '"mode"' in line:
                violations.append(f"{path.relative_to(ROOT).as_posix()}:{line_number}")

    return tuple(violations)


def test_mode_terms_stay_out_of_non_orchestrator_service_source() -> None:
    # Given: mic/asr/comments/tts/sound are mode-agnostic services.

    # When: their source trees are scanned for mode policy terms.

    """函数契约说明.

    功能: 验证 mode terms stay out of non
    orchestrator service source
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    violations = service_source_violations()

    # Then: only Orchestrator owns mode-specific policy language.

    assert violations == ()


def test_mode_specific_terms_stay_out_of_mode_agnostic_schemas() -> None:
    # Given: shared schemas may name generic session/LLM mode fields only.

    # When: schemas are scanned for concrete mode policy terms and mode on IO events.

    """函数契约说明.

    功能: 验证 mode specific terms stay out
    of mode agnostic schemas
    的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    violations = schema_mode_violations()

    # Then: schema fields for ASR/comments/TTS/sound stay mode-agnostic.

    assert violations == ()


def test_isolation_scanner_flags_mode_specific_service_source() -> None:
    # Given: a synthetic mode-aware service source violation.

    """函数契约说明.

    功能: 验证 isolation scanner flags mode
    specific service source 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    text = "LECTURER_MODE = 'lecturer'\n"

    # When: the same pattern used for source scanning evaluates it.

    violation = SERVICE_MODE_PATTERN.search(text)

    # Then: the contract would fail if a non-Orchestrator service added this term.

    assert violation is not None


def test_isolation_scanner_flags_mode_specific_schema_term() -> None:
    # Given: a synthetic ASR schema line that leaks a concrete mode.

    """函数契约说明.

    功能: 验证 isolation scanner flags mode
    specific schema term 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    text = '"asr.final": {"required": ["text", "lecturer"]}'

    # When: the same pattern used for schema scanning evaluates it.

    violation = SCHEMA_MODE_PATTERN.search(text)

    # Then: the contract would fail if an IO schema added this term.

    assert violation is not None
