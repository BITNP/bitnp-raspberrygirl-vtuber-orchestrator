
from pathlib import Path

import pytest

from orchestrator.ids import SessionId, TraceId
from orchestrator.interaction_ingress import SessionInteractionIngress
from orchestrator.interactions import InteractionAccepted
from orchestrator.retrieval import ReadonlyLlamaIndexProvider
from orchestrator.sessions import EventCorrelation, EventSequence, SessionScheduler


def test_production_ingress_routes_comment_through_scheduler_reducer() -> None:
    # Given: one production ingress composed with its session scheduler controls.


    scheduler = SessionScheduler(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
    )

    ingress = SessionInteractionIngress.create(scheduler)

    # When: comments ingress submits a normalized, correlated audience proposal.

    outcome = ingress.receive_comment(
        text="解释量化",
        correlation=EventCorrelation(
            trace_id=TraceId("trace-1"),
            session_id=SessionId("session-1"),
            sequence=EventSequence(1),
        ),
    )

    # Then: the real reducer opens the scheduler's first monotonic turn.

    assert outcome == InteractionAccepted(turn_id="turn-0001")

    assert scheduler.event_history[0].event.event_type == "audience.input"


def test_control_envelope_routes_comments_but_leaves_media_for_transport() -> None:
    # Given: a live ingress sharing one session scheduler with transport.


    scheduler = SessionScheduler(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
    )

    ingress = SessionInteractionIngress.create(scheduler)

    # When: a comments envelope and a media envelope reach the common listener.

    comment = (
        '{"event_type":"audience.input","source":"comments","trace_id":"trace-1",'
        '"session_id":"session-1","seq":1,"data":{"text":"解释量化"}}'
    )

    media = '{"event_type":"media.rtp.sink.ready","source":"sound","data":{}}'

    # Then: only the typed comments proposal opens a reducer-controlled turn.

    assert ingress.receive_control(comment) is True

    assert ingress.receive_control(media) is False

    assert scheduler.snapshot.active_turn_id is not None


def test_duplicate_correlated_comment_opens_only_one_turn() -> None:
    # Given: one live comments frame and its exact transport replay.


    scheduler = SessionScheduler(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
    )

    ingress = SessionInteractionIngress.create(scheduler)

    comment = (
        '{"event_type":"audience.input","source":"comments","trace_id":"trace-1",'
        '"session_id":"session-1","seq":1,"data":{"text":"解释量化"}}'
    )

    # When: the same trace/session/sequence envelope is received twice.

    first = ingress.receive_control(comment)

    duplicate = ingress.receive_control(comment)

    # Then: exactly one reducer-controlled turn is materialized.

    assert (first, duplicate) == (True, True)

    assert [event.turn_id for event in scheduler.event_history] == ["turn-0001"]


def test_ingress_uses_configured_readonly_knowledge_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given: deployment explicitly supplies the controlled, startup-only corpus.

    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    _ = (knowledge / "product.md").write_text("树莓女孩", encoding="utf-8")
    monkeypatch.setenv("ORCHESTRATOR_KNOWLEDGE_DIR", str(knowledge))
    monkeypatch.setenv("ORCHESTRATOR_STATE_DIR", str(tmp_path / "state"))
    scheduler = SessionScheduler(
        session_id=SessionId("session-knowledge"), turn_id_prefix="turn"
    )

    # When: the production ingress is built for the session.

    ingress = SessionInteractionIngress.create(scheduler)

    # Then: it owns an immutable LlamaIndex-core corpus, never a fixture corpus.

    assert isinstance(ingress.data.retrieval, ReadonlyLlamaIndexProvider)
