from __future__ import annotations

import json
from base64 import b64decode, b64encode

import pytest
from cryptography.exceptions import InvalidTag

from orchestrator.control_ingress import ProfileEnrollmentControl
from orchestrator.identity import VoiceProfileId
from orchestrator.ids import SessionId, TraceId
from orchestrator.json_boundary import parse_json_value
from orchestrator.scheduler_runtime import SessionRuntime
from orchestrator.sessions import EventCorrelation, EventSequence
from orchestrator.task_registry import SchedulerTaskConfig, TaskKind
from orchestrator.transport_control import EnvelopeCorrelation, VoiceEvidence
from orchestrator.voice_templates import (
    DecryptedVoiceTemplate,
    VoiceTemplateProtector,
    match_voice,
    modular_intervals_overlap,
)


def test_voice_template_aead_binds_session_profile_model_and_detects_tamper() -> None:
    protector = VoiceTemplateProtector(bytes(range(32)))
    profile_id = VoiceProfileId("profile-1")
    protected = protector.encrypt(
        session_id="session-1",
        profile_id=profile_id,
        model_revision="campp-v1",
        embedding=(3.0, 4.0),
    )
    decrypted = protector.decrypt(
        session_id="session-1", profile_id=profile_id, template=protected
    )
    assert decrypted.model_revision == "campp-v1"
    assert decrypted.embedding == pytest.approx((0.6, 0.8))

    envelope = parse_json_value(protected.ciphertext.decode())
    assert isinstance(envelope, dict)
    encoded_ciphertext = envelope["ciphertext"]
    assert isinstance(encoded_ciphertext, str)
    ciphertext = bytearray(b64decode(encoded_ciphertext))
    ciphertext[0] ^= 1
    envelope["ciphertext"] = b64encode(ciphertext).decode()
    tampered = type(protected)(json.dumps(envelope).encode())
    with pytest.raises(InvalidTag):
        _ = protector.decrypt(
            session_id="session-1", profile_id=profile_id, template=tampered
        )


def test_voice_match_rejects_ambiguous_below_threshold_and_model_mismatch() -> None:
    first = VoiceProfileId("first")
    second = VoiceProfileId("second")
    assert match_voice(
        "v1",
        (1.0, 0.0),
        {
            first: DecryptedVoiceTemplate("v1", (1.0, 0.0)),
            second: DecryptedVoiceTemplate("v1", (0.999, 0.001)),
        },
    ).profile_id is None
    assert match_voice(
        "v2", (1.0, 0.0), {first: DecryptedVoiceTemplate("v1", (1.0, 0.0))}
    ).profile_id is None
    assert match_voice(
        "v1", (1.0, 0.0), {first: DecryptedVoiceTemplate("v1", (0.8, 0.6))}
    ).profile_id is None


def test_evidence_enrollment_is_recent_session_local_and_single_use() -> None:
    clock = [1_000]
    runtime = SessionRuntime.create(
        session_id=SessionId("session-1"),
        turn_id_prefix="turn",
        task_config=SchedulerTaskConfig(frozenset(TaskKind), 2),
        clock=lambda: clock[0],
    )
    runtime.configure_voice_identity(bytes(range(32)), evidence_ttl_seconds=120)
    evidence = VoiceEvidence(
        session_id="session-1",
        evidence_id="evidence-1",
        stream_id="mic-1",
        input_epoch=1,
        rtp_start_timestamp=0xFFFF_FF00,
        rtp_end_timestamp=0x0000_0180,
        embedding_model_revision="campp-v1",
        embedding=(1.0, 0.0),
        speech_ms=640,
        quality_score=0.9,
        correlation=EnvelopeCorrelation("trace", "session-1", 1),
    )
    assert runtime.receive_voice_evidence(evidence)
    control = ProfileEnrollmentControl(
        profile_id=VoiceProfileId("profile-1"),
        preferred_name="嘉宾",
        evidence_id="evidence-1",
        consented=True,
        correlation=EventCorrelation(
            TraceId("operator"), SessionId("session-1"), EventSequence(2)
        ),
    )
    assert runtime.enroll_profile_from_evidence(control).accepted
    assert not runtime.enroll_profile_from_evidence(control).accepted
    assert modular_intervals_overlap(
        0xFFFF_FF00, 0x0000_0180, 0x0000_0000, 0x0000_0100
    )
