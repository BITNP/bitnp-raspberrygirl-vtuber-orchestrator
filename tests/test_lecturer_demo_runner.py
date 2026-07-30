"""模块契约说明.

职责: 为测试场景提供断言、夹具和回归用例。
契约: 模块只提供注释所描述的公开入口,不在文档更新中改变运行时行为。
"""

from __future__ import annotations

import ast
import runpy
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Final

import pytest

from orchestrator.json_boundary import JsonValue, parse_json_value

ROOT: Final = Path(__file__).resolve().parents[1]

SCRIPT: Final = ROOT / "scripts" / "run_lecturer_demo.py"

SAMPLE: Final = ROOT / "samples" / "lecturer" / "bitnet_intro_zh.json"


def test_demo_uses_no_independent_media_package_imports() -> None:
    # Given: the migrated demo source files owned by the Orchestrator repository.

    """函数契约说明.

    功能: 验证 demo uses no independent
    media package imports 的回归场景和可观察结果。
    参数: 无显式业务参数。
    契约: 同步调用。 返回 `None`。
    """

    demo_sources = (
        ROOT / "scripts" / "lecturer_demo_lib.py",
        ROOT / "scripts" / "lecturer_demo_protocol.py",
    )

    # When: their direct imports are inspected without executing the demo.

    imported_roots = {
        alias.name.split(".", 1)[0]
        for source in demo_sources
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".", 1)[0]
        for source in demo_sources
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    # Then: demo serialization uses local Orchestrator fakes, not media packages.

    assert imported_roots & {"asr", "tts"} == set()


def test_demo_runner_writes_protocol_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the sample Chinese lecturer script and an evidence output path.

    """函数契约说明.

    功能: 验证 demo runner writes protocol
    evidence 的回归场景和可观察结果。
    参数: tmp_path: Path。 必填。 monkeypatch:
    pytest.MonkeyPatch。 必填。
    契约: 同步调用。 返回 `None`。
    """

    evidence = tmp_path / "lecturer-demo.json"

    monkeypatch.setattr(sys, "path", [str(ROOT / "scripts"), *sys.path])

    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "--script", str(SAMPLE), "--evidence", str(evidence)],
    )

    stdout = StringIO()

    # When: the real demo script entrypoint is executed.

    with redirect_stdout(stdout), pytest.raises(SystemExit) as exit_info:
        _run_demo_script(SCRIPT)

    # Then: the command reports success and writes protocol-shaped evidence.

    assert exit_info.value.code == 0

    assert "LECTURER DEMO PASSED" in stdout.getvalue()

    data = parse_json_value(evidence.read_text(encoding="utf-8"))

    assert isinstance(data, dict)

    event_types = _event_types(data)

    assert "media.stream.command" in event_types

    assert "media.stream.state" in event_types

    assert "vtuber.caption.command" in event_types

    assert "vtuber.expression.command" in event_types

    assert "vtuber.action.command" in event_types

    assert "vtuber.scene.command" in event_types

    topology = data["topology"]

    assert isinstance(topology, dict)

    assert topology["peer_edges"] == []

    assert data["status"] == "passed"

    assert data["script_title"] == "BitNet 讲解模式演示"


def _event_types(data: dict[str, JsonValue]) -> tuple[str, ...]:
    """函数契约说明.

    功能: 执行 _event_types 的同步逻辑,并协调
    isinstance, tuple, append。
    参数: data: dict[str, JsonValue]。 必填。
    契约: 同步调用。 返回 `tuple[str, ...]`。
    """

    events = data["events"]

    assert isinstance(events, list)

    event_types: list[str] = []

    for event in events:
        assert isinstance(event, dict)

        event_type = event["event_type"]

        assert isinstance(event_type, str)

        event_types.append(event_type)

    return tuple(event_types)


def _run_demo_script(script: Path) -> None:
    """函数契约说明.

    功能: 执行 _run_demo_script 的同步逻辑,并协调
    run_path, str。
    参数: script: Path。 必填。
    契约: 同步调用。 返回 `None`。
    """

    module_globals = runpy.run_path(str(script), run_name="__main__")

    assert module_globals["__name__"] == "__main__"
