from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lecturer_demo_protocol import JsonObject, event, is_peer_edge  # noqa: E402
from orchestrator.lecturer_script import LectureStep, parse_lecture_script  # noqa: E402


@dataclass(frozen=True, slots=True)
class DemoStepRun:
    index: int
    step: LectureStep


def run_demo(script_path: Path) -> JsonObject:
    lecture = parse_lecture_script(script_path)
    events: list[JsonObject] = [
        event(
            event_type="session.created",
            source="orchestrator",
            seq=1,
            data={"created_at": "2026-07-14T00:00:00Z", "mode": "lecturer"},
        ),
    ]
    edges: list[JsonObject] = []
    for index, step in enumerate(lecture.steps, start=1):
        _append_step(events, edges, DemoStepRun(index, step))
    return {
        "status": "passed",
        "script_title": lecture.title,
        "events": events,
        "topology": {
            "edges": edges,
            "peer_edges": [edge for edge in edges if is_peer_edge(edge)],
        },
    }


def write_demo_evidence(script_path: Path, evidence_path: Path) -> None:
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(run_demo(script_path), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_step(
    events: list[JsonObject],
    edges: list[JsonObject],
    run: DemoStepRun,
) -> None:
    turn_id = f"turn-demo-{run.index:04d}"
    segment_id = f"seg-demo-{run.index:04d}"
    stream_id = f"rtp-demo-{run.index:04d}"
    seq = len(events) + 1
    records = (
        (
            "media.stream.command",
            "orchestrator",
            {"command_id": f"command-{stream_id}", "stream_id": stream_id, "start_rtp_timestamp": 0},
        ),
        (
            "media.stream.state",
            "sound",
            {
                "stream_id": stream_id,
                "state": "finished",
                "playback_position_ms": 120,
            },
        ),
        (
            "vtuber.caption.command",
            "orchestrator",
            {
                "caption_id": f"caption-{segment_id}",
                "text": run.step.narration,
                "audio_stream_id": stream_id,
                "start_at_ms": 0,
                "end_at_ms": 120,
                "start_rtp_timestamp": 0,
                "end_rtp_timestamp": 2_880,
            },
        ),
        (
            "vtuber.expression.command",
            "orchestrator",
            {
                "expression_id": f"expression-{segment_id}",
                "expression": run.step.expression,
                "audio_stream_id": stream_id,
                "start_at_ms": 0,
                "end_at_ms": 120,
                "start_rtp_timestamp": 0,
                "end_rtp_timestamp": 2_880,
            },
        ),
        (
            "vtuber.action.command",
            "orchestrator",
            {
                "action_id": f"action-{segment_id}",
                "action": run.step.action,
                "audio_stream_id": stream_id,
                "start_at_ms": 0,
                "end_at_ms": 120,
                "start_rtp_timestamp": 0,
                "end_rtp_timestamp": 2_880,
            },
        ),
        (
            "vtuber.scene.command",
            "orchestrator",
            {
                "scene_id": f"scene-{segment_id}",
                "scene": run.step.scene,
                "slide_id": run.step.slide.id,
                "slide_title": run.step.slide.title,
                "page": run.step.slide.page,
            },
        ),
    )
    for event_type, source, data in records:
        events.append(
            event(
                event_type=event_type,
                source=source,
                seq=seq,
                data=data,
                turn_id=turn_id,
                segment_id=segment_id,
            ),
        )
        seq += 1
    edges.extend(
        (
            {
                "source": "orchestrator",
                "target": "sound",
                "event_type": "media.stream.command",
            },
            {
                "source": "sound",
                "target": "orchestrator",
                "event_type": "media.stream.state",
            },
            {
                "source": "orchestrator",
                "target": "frontend",
                "event_type": "vtuber.caption.command",
            },
            {
                "source": "orchestrator",
                "target": "frontend",
                "event_type": "vtuber.expression.command",
            },
            {
                "source": "orchestrator",
                "target": "frontend",
                "event_type": "vtuber.action.command",
            },
            {
                "source": "orchestrator",
                "target": "frontend",
                "event_type": "vtuber.scene.command",
            },
        ),
    )
