"""Structured semantic interruption policy for an active generated answer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, Protocol, final, override

from orchestrator.json_boundary import JsonBoundaryError, parse_json_value
from orchestrator.llm import CancellationToken
from orchestrator.provider_streaming import ProviderResponseError
from orchestrator.streaming_contracts import (
    CancellationEpoch,
    FlushAcknowledgement,
    FlushAdmission,
    FlushClock,
    FlushRequestId,
    FlushSender,
    GeneratedSsrc,
    SegmentId,
    StreamFlush,
    StreamKey,
    TurnId,
)

if TYPE_CHECKING:
    from orchestrator.observability import OnsiteObservability, OnsiteStage

CLASSIFIER_TIMEOUT_MS: Final = 400
_MAX_ANSWER_EXCERPT_CHARS: Final = 512


@dataclass(frozen=True, slots=True)
class EndpointedTranscript:
    """Stable ASR final eligible for semantic interruption evaluation."""

    stream: StreamKey
    text: str
    turn_id: TurnId
    segment_id: SegmentId


@dataclass(frozen=True, slots=True)
class ActiveAnswer:
    """Playback identity and bounded context for one generated answer."""

    stream: StreamKey
    turn_id: TurnId
    segment_id: SegmentId
    cancellation_epoch: CancellationEpoch
    answer_excerpt: str
    target_generated_ssrc: GeneratedSsrc
    cancellation: CancellationToken = field(default_factory=CancellationToken)


@dataclass(frozen=True, slots=True)
class BargeInClassifierRequest:
    """Structured input passed to the dedicated classifier boundary."""

    transcript: str
    active_turn_id: TurnId
    active_segment_id: SegmentId
    active_answer_excerpt: str
    timeout_ms: Literal[400] = CLASSIFIER_TIMEOUT_MS


class BargeInClassifier(Protocol):
    """Returns one externally produced structured classifier response."""

    def classify(self, request: BargeInClassifierRequest) -> str:
        """Evaluate interruption intent within the request's stated deadline."""
        ...


@dataclass(frozen=True, slots=True)
class BargeInClassifierFailure:
    """Correlated degraded-decision record for one active answer."""

    stream: StreamKey
    turn_id: TurnId
    segment_id: SegmentId
    reason: Literal["timeout", "malformed", "unavailable"]


@dataclass(frozen=True, slots=True)
class BargeInCancellation:
    """Cancellation intent for provider, TTS, and generated RTP owners."""

    turn_id: TurnId
    segment_id: SegmentId
    targets: tuple[Literal["llm", "tts", "rtp"], ...] = ("llm", "tts", "rtp")


@dataclass(frozen=True, slots=True)
class _Decision:
    utterance: EndpointedTranscript
    active: ActiveAnswer | None
    value: Literal["interrupt", "continue"]


@final
class SemanticBargeInGate:
    """Preserve playback unless an exact semantic interrupt reaches Sound admission."""

    def __init__(
        self, *, classifier: BargeInClassifier, clock: FlushClock, sender: FlushSender
    ) -> None:
        """Bind the classifier and the existing Sound flush-admission boundary."""
        self._classifier = classifier
        self._flush_admission = FlushAdmission(clock=clock, sender=sender)
        self._active: ActiveAnswer | None = None
        self._queued: EndpointedTranscript | None = None
        self._replacement: EndpointedTranscript | None = None
        self._replacement_flush: StreamFlush | None = None
        self._epoch = CancellationEpoch(0)
        self._flush_sequence = 0
        self._failures: list[BargeInClassifierFailure] = []
        self._cancellations: list[BargeInCancellation] = []
        self._observability: OnsiteObservability | None = None

    def set_observability(self, observability: OnsiteObservability) -> None:
        """Attach the shared onsite recorder after stream composition."""
        self._observability = observability

    @property
    def cancellation_epoch(self) -> CancellationEpoch:
        """Expose the current epoch without advancing it for a continue decision."""
        return self._epoch

    @property
    def failures(self) -> tuple[BargeInClassifierFailure, ...]:
        """Return correlated classifier fallback records."""
        return tuple(self._failures)

    @property
    def cancellations(self) -> tuple[BargeInCancellation, ...]:
        """Return cancellation intents emitted only for semantic interrupts."""
        return tuple(self._cancellations)

    def activate(self, active: ActiveAnswer) -> None:
        """Set the active answer whose playback may be semantically interrupted."""
        self._active = active
        self._epoch = active.cancellation_epoch

    def handle(self, utterance: EndpointedTranscript) -> None:
        """Classify and apply a stable transcript against the current active answer."""
        self.apply(self.classify(utterance))

    def classify(self, utterance: EndpointedTranscript) -> _Decision:
        """Return an exact decision or a correlated continue fallback."""
        active = self._active
        if active is None or active.stream != utterance.stream:
            return _Decision(utterance=utterance, active=None, value="continue")
        request = BargeInClassifierRequest(
            transcript=utterance.text,
            active_turn_id=active.turn_id,
            active_segment_id=active.segment_id,
            active_answer_excerpt=active.answer_excerpt[-_MAX_ANSWER_EXCERPT_CHARS:],
        )
        try:
            value = _parse_decision(self._classifier.classify(request))
        except TimeoutError:
            self._record_failure(active, "timeout")
            value = "continue"
        except (JsonBoundaryError, BargeInResponseError):
            self._record_failure(active, "malformed")
            value = "continue"
        except (OSError, ProviderResponseError):
            self._record_failure(active, "unavailable")
            value = "continue"
        self._record("classifier_decision", active)
        return _Decision(utterance=utterance, active=active, value=value)

    def apply(self, decision: _Decision) -> None:
        """Apply a decision only while the evaluated answer remains active."""
        if self._active != decision.active:
            return
        match decision.value:
            case "continue":
                self._queued = decision.utterance
            case "interrupt":
                if decision.active is not None:
                    self._interrupt(decision)

    def acknowledge(self, acknowledgement: FlushAcknowledgement) -> None:
        """Forward the exact Sound acknowledgement to replacement admission."""
        _ = self._flush_admission.acknowledge(acknowledgement)

    def pop_queued_utterance(self) -> EndpointedTranscript | None:
        """Return the newest continued utterance once for later turn processing."""
        queued = self._queued
        self._queued = None
        return queued

    def pop_admitted_replacement(self) -> EndpointedTranscript | None:
        """Return the replacement only after its exact Sound flush acknowledgement."""
        replacement = self._replacement
        replacement_flush = self._replacement_flush
        if (
            replacement is None
            or replacement_flush is None
            or not self._flush_admission.admitted(replacement_flush)
        ):
            return None
        self._replacement = None
        self._replacement_flush = None
        return replacement

    def _interrupt(self, decision: _Decision) -> None:
        active = decision.active
        if active is None:
            return
        self._epoch = CancellationEpoch(int(active.cancellation_epoch) + 1)
        _ = active.cancellation.cancel(reason="semantic_interrupt")
        self._cancellations.append(
            BargeInCancellation(turn_id=active.turn_id, segment_id=active.segment_id)
        )
        self._record("cancellation", active)
        self._replacement = decision.utterance
        self._flush_sequence += 1
        flush = StreamFlush(
            stream=active.stream,
            turn_id=active.turn_id,
            segment_id=active.segment_id,
            cancellation_epoch=self._epoch,
            request_id=FlushRequestId(
                f"{active.stream.session_id}:{active.stream.stream_id}:flush:{self._flush_sequence}"
            ),
            target_generated_ssrc=active.target_generated_ssrc,
        )
        self._replacement_flush = flush
        self._flush_admission.begin(flush)
        self._record("flush", active)
        self._active = None

    def _record_failure(
        self,
        active: ActiveAnswer,
        reason: Literal["timeout", "malformed", "unavailable"],
    ) -> None:
        self._failures.append(
            BargeInClassifierFailure(
                stream=active.stream,
                turn_id=active.turn_id,
                segment_id=active.segment_id,
                reason=reason,
            )
        )
        self._record("classifier_failure", active)

    def _record(self, stage: OnsiteStage, active: ActiveAnswer) -> None:
        observability = self._observability
        if observability is not None:
            correlation = observability.correlation(
                active.stream,
                str(active.turn_id),
                str(active.segment_id),
                active.cancellation_epoch,
            )
            if correlation is not None:
                observability.record(stage, correlation)


@dataclass(frozen=True, slots=True)
class BargeInResponseError(ValueError):
    """Raised when a classifier response is not the exact structured contract."""

    @override
    def __str__(self) -> str:
        """Render the typed malformed-response failure."""
        return "semantic barge-in classifier returned malformed output"


def _parse_decision(response: str) -> Literal["interrupt", "continue"]:
    payload = parse_json_value(response)
    if not isinstance(payload, dict) or set(payload) != {"decision"}:
        raise BargeInResponseError
    value = payload["decision"]
    match value:
        case "interrupt":
            return "interrupt"
        case "continue":
            return "continue"
        case _:
            raise BargeInResponseError
