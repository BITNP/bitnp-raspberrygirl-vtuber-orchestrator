from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Literal, Protocol, cast, final, override

from orchestrator.streaming_contracts import (
    CancellationEpoch,
    FlushAcknowledgement,
    SegmentId,
    StreamFlush,
    StreamKey,
)
from orchestrator.transport_control import (
    AsrFinal,
    AsrPartial,
    ControlEvent,
    EnvelopeCorrelation,
    MicInputRegistration,
    SinkRegistration,
    StreamReady,
    StreamState,
    VoiceEvidence,
    parse_control_event,
)
from orchestrator.tts_rtp import generated_ssrc

_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from orchestrator.ids import ConnectionId
    from orchestrator.observability import OnsiteObservability, OnsiteStage
    from orchestrator.scheduler_reflex import OutputLease, SchedulerOutputFence
    from orchestrator.task_registry import TaskId


type PeerAddress = tuple[str, int]


RTP_HEADER_BYTES = 12

L16_FRAME_BYTES = 640

RTP_V2_HEADER = 0x80

RTP_PAYLOAD_TYPE = 96


class DatagramSender(Protocol):
    def sendto(self, data: bytes, addr: PeerAddress) -> None: ...

    def close(self) -> None: ...


class OnsiteBridge(Protocol):
    def set_output_callback(
        self,
        callback: Callable[[StreamKey, CancellationEpoch, bytes], Awaitable[None]],
    ) -> None: ...

    def set_output_finished_callback(
        self, callback: Callable[[StreamKey, CancellationEpoch], Awaitable[None]]
    ) -> None: ...

    def set_output_authorizer(
        self,
        callback: Callable[[StreamKey, CancellationEpoch], bool],
    ) -> None: ...

    def set_response_output_preparer(
        self,
        callback: Callable[
            [StreamKey, CancellationEpoch, str], Awaitable[CancellationEpoch | None]
        ],
    ) -> None: ...

    def set_replacement_callback(
        self,
        callback: Callable[[StreamKey, SegmentId], Awaitable[CancellationEpoch | None]],
    ) -> None: ...

    def invalidate_stream(
        self, stream: StreamKey, next_epoch: CancellationEpoch
    ) -> None: ...

    async def wait_quiescent(self) -> None: ...


@dataclass(frozen=True, slots=True)
class DuplicateRouteError(Exception):
    stream: StreamKey

    @override
    def __str__(self) -> str:
        return f"duplicate RTP route: {self.stream.session_id}/{self.stream.stream_id}"


@final
class RtpHub:
    def __init__(
        self,
        transport: DatagramSender | None = None,
        *,
        onsite_bridge: OnsiteBridge | None = None,
    ) -> None:
        self._transport: DatagramSender | None = transport

        self._onsite_bridge: OnsiteBridge | None = onsite_bridge

        self._output_fence: SchedulerOutputFence | None = None

        self._output_fences: dict[str, SchedulerOutputFence] = {}

        self._observability: OnsiteObservability | None = None

        self._correlations: dict[StreamKey, EnvelopeCorrelation] = {}

        self._mic_inputs: set[StreamKey] = set()

        self._mic_input_owners: dict[StreamKey, ConnectionId] = {}

        self._sinks: dict[StreamKey, PeerAddress] = {}

        self._sink_owners: dict[StreamKey, ConnectionId] = {}

        self._route_generations: dict[StreamKey, int] = {}

        self._output_command_callback: (
            Callable[[StreamKey, int], Awaitable[None]] | None
        ) = None

        self._output_command_tasks: set[asyncio.Future[None]] = set()

        self._replacement_flush_callback: (
            Callable[[StreamFlush], Awaitable[None]] | None
        ) = None

        self._replacement_admit_callback: (
            Callable[[OutputLease, StreamFlush], Awaitable[bool]] | None
        ) = None

        # Production replacement exchanges are owned by SessionRuntime tasks.
        # Kept optional for standalone transport-contract tests and for routes
        # which have not yet been attached to a session runtime.
        self._replacement_task_schedule: (
            Callable[
                [StreamKey, OutputLease, StreamFlush],
                TaskId | Literal[False] | None,
            ]
            | None
        ) = None
        self._replacement_task_is_current: (
            Callable[[StreamKey, TaskId], bool] | None
        ) = None
        self._replacement_task_complete: (
            Callable[[StreamKey, TaskId], bool] | None
        ) = None
        self._replacement_task_fail: (
            Callable[[StreamKey, TaskId, str], None] | None
        ) = None

        self._voice_evidence_callbacks: dict[
            str, Callable[[VoiceEvidence], bool]
        ] = {}

        self._last_asr_sequences: dict[StreamKey, int] = {}

        self._rtp_egress_counts: dict[tuple[StreamKey, CancellationEpoch], int] = {}

        if onsite_bridge is not None:
            onsite_bridge.set_output_callback(self.deliver_generated_rtp)
            replacement_callback = cast(
                "Callable[[Callable[[StreamKey, SegmentId], Awaitable[CancellationEpoch | None]]], None] | None",  # noqa: E501
                getattr(onsite_bridge, "set_replacement_callback", None),
            )
            if replacement_callback is not None:
                replacement_callback(self.begin_onsite_replacement)

    def set_output_finished_callback(
        self, callback: Callable[[StreamKey, CancellationEpoch], Awaitable[None]]
    ) -> None:
        bridge = self._onsite_bridge
        if bridge is not None:
            bridge.set_output_finished_callback(callback)

    def set_output_command_callback(
        self, callback: Callable[[StreamKey, int], Awaitable[None]]
    ) -> None:
        self._output_command_callback = callback

    def set_replacement_callbacks(
        self,
        request_flush: Callable[[StreamFlush], Awaitable[None]],
        admit_replacement: Callable[[OutputLease, StreamFlush], Awaitable[bool]],
    ) -> None:
        self._replacement_flush_callback = request_flush
        self._replacement_admit_callback = admit_replacement

    def set_replacement_task_callbacks(
        self,
        schedule: Callable[
            [StreamKey, OutputLease, StreamFlush], TaskId | Literal[False] | None
        ],
        is_current: Callable[[StreamKey, TaskId], bool],
        complete: Callable[[StreamKey, TaskId], bool],
        fail: Callable[[StreamKey, TaskId, str], None],
    ) -> None:
        """Install the scheduler-owned lifecycle for replacement flushes."""
        self._replacement_task_schedule = schedule
        self._replacement_task_is_current = is_current
        self._replacement_task_complete = complete
        self._replacement_task_fail = fail

    def set_voice_evidence_callback(
        self, session_id: str, callback: Callable[[VoiceEvidence], bool]
    ) -> None:
        self._voice_evidence_callbacks[session_id] = callback

    def accept_asr_final(
        self, event: AsrFinal, owner: ConnectionId | None = None
    ) -> bool:
        """Validate that a final belongs to a live Mic route exactly once.

        Recognition text is deliberately not accepted from RTP or arbitrary
        peers: it must arrive on the authenticated control socket after Mic's
        source registration and be for the route's active cancellation epoch.
        """
        stream = StreamKey(event.session_id, event.stream_id)
        if stream not in self._mic_inputs:
            return False
        if owner is not None and self._mic_input_owners.get(stream) != owner:
            return False
        if int(event.cancellation_epoch) != self._route_generations.get(stream, 0):
            return False
        previous = self._last_asr_sequences.get(stream)
        if previous is not None and event.correlation.seq <= previous:
            return False
        self._last_asr_sequences[stream] = event.correlation.seq
        return True

    async def begin_onsite_replacement(
        self, stream: StreamKey, segment_id: SegmentId
    ) -> CancellationEpoch | None:
        """Prepare a replacement only after its first audio frame is available.

        The old lease remains eligible until Sound acknowledges the flush.  The
        caller holds the new frame locally while this method waits, so no new
        RTP can be dropped into the pending-fence gap.
        """
        output_fence = self._fence_for(stream)
        correlation = self._correlations.get(stream)
        request_flush = self._replacement_flush_callback
        admit_replacement = self._replacement_admit_callback
        if (
            output_fence is None
            or correlation is None
            or request_flush is None
            or admit_replacement is None
        ):
            return None
        try:
            replacement, flush = output_fence.interrupt(
                stream=stream, segment_id=segment_id, correlation=correlation
            )
        except (KeyError, RuntimeError):
            return None
        schedule = self._replacement_task_schedule
        if schedule is not None:
            scheduled = schedule(stream, replacement, flush)
            if scheduled is False:
                _ = output_fence.abandon_replacement(stream)
                return None
        else:
            scheduled = None
        return await self._await_replacement_flush(
            stream, replacement, flush, scheduled
        )

    async def _await_replacement_flush(
        self,
        stream: StreamKey,
        replacement: OutputLease,
        flush: StreamFlush,
        task_id: TaskId | None,
    ) -> CancellationEpoch | None:
        """Wait for one exact Sound acknowledgement behind a task result fence."""
        # The caller captured both collaborators before creating the pending
        # replacement, so this exchange has a stable control surface.
        output_fence = cast("SchedulerOutputFence", self._fence_for(stream))
        request_flush = cast(
            "Callable[[StreamFlush], Awaitable[None]]",
            self._replacement_flush_callback,
        )
        admitted = False
        reason = "sound_flush_timeout"
        try:
            if self._replacement_task_current(stream, task_id):
                _LOGGER.debug(
                    "sound_flush_request session=%s stream=%s turn=%s %s",
                    stream.session_id,
                    stream.stream_id,
                    replacement.turn_id,
                    (
                        f"segment={replacement.segment_id} "
                        f"epoch={replacement.cancellation_epoch}"
                    ),
                )
                await request_flush(flush)
            else:
                reason = "sound_flush_stale_before_request"
            deadline = monotonic() + 3.0
            while reason == "sound_flush_timeout" and monotonic() < deadline:
                if not self._replacement_task_current(stream, task_id):
                    reason = "sound_flush_stale"
                    break
                if output_fence.can_emit(stream, replacement.cancellation_epoch):
                    _LOGGER.debug(
                        "sound_flush_acknowledged session=%s stream=%s turn=%s %s",
                        stream.session_id,
                        stream.stream_id,
                        replacement.turn_id,
                        (
                            f"segment={replacement.segment_id} "
                            f"epoch={replacement.cancellation_epoch}"
                        ),
                    )
                    result = await self._admit_replacement_if_current(
                        stream, replacement, flush, task_id
                    )
                    if result is None and output_fence.commit_replacement(
                        stream, replacement.cancellation_epoch
                    ):
                        admitted = True
                    else:
                        reason = result or "sound_flush_result_rejected"
                    break
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            self._fail_replacement_task(stream, task_id, "sound_flush_cancelled")
            _ = output_fence.abandon_replacement(stream)
            raise
        except OSError:
            reason = "sound_flush_transport_failed"
        if admitted:
            return replacement.cancellation_epoch
        self._fail_replacement_task(stream, task_id, reason)
        _ = output_fence.abandon_replacement(stream)
        return None

    async def _admit_replacement_if_current(
        self,
        stream: StreamKey,
        replacement: OutputLease,
        flush: StreamFlush,
        task_id: TaskId | None,
    ) -> str | None:
        if not self._replacement_task_current(stream, task_id):
            return "sound_flush_stale_before_admission"
        admit = self._replacement_admit_callback
        if admit is None or not await admit(replacement, flush):
            return "sound_flush_admission_rejected"
        if not self._complete_replacement_task(stream, task_id):
            return "sound_flush_result_rejected"
        return None

    def _replacement_task_current(
        self, stream: StreamKey, task_id: TaskId | None
    ) -> bool:
        current = self._replacement_task_is_current
        return task_id is None or current is None or current(stream, task_id)

    def _complete_replacement_task(
        self, stream: StreamKey, task_id: TaskId | None
    ) -> bool:
        complete = self._replacement_task_complete
        return task_id is None or complete is None or complete(stream, task_id)

    def _fail_replacement_task(
        self, stream: StreamKey, task_id: TaskId | None, reason: str
    ) -> None:
        fail = self._replacement_task_fail
        if task_id is not None and fail is not None:
            fail(stream, task_id, reason)

    def attach_transport(self, transport: DatagramSender) -> None:
        self._transport = transport

    def set_observability(self, observability: OnsiteObservability) -> None:
        self._observability = observability

    def set_output_fence(
        self, output_fence: SchedulerOutputFence, session_id: str | None = None
    ) -> None:
        if session_id is None:
            self._output_fence = output_fence
        else:
            self._output_fences[session_id] = output_fence

        bridge = self._onsite_bridge

        if bridge is not None:
            bridge.set_output_authorizer(self.authorize_onsite_output)
            bridge.set_response_output_preparer(
                self.prepare_onsite_response_output
            )

    def _fence_for(self, stream: StreamKey) -> SchedulerOutputFence | None:
        return self._output_fences.get(stream.session_id, self._output_fence)

    def authorize_onsite_output(
        self, stream: StreamKey, epoch: CancellationEpoch
    ) -> bool:
        """Activate a scheduler lease for a finalized onsite utterance."""
        output_fence = self._fence_for(stream)

        if output_fence is None:
            return True

        correlation = self._correlations.get(stream)

        if correlation is None:
            return False

        lease = output_fence.activate(
            stream=stream,
            segment_id=SegmentId(f"onsite-{stream.stream_id}-{int(epoch)}"),
            correlation=correlation,
        )

        callback = self._output_command_callback
        if callback is not None:
            task = asyncio.ensure_future(
                callback(stream, int(lease.cancellation_epoch))
            )
            self._output_command_tasks.add(task)
            task.add_done_callback(self._output_command_tasks.discard)

        return lease.cancellation_epoch == epoch

    async def allocate_onsite_response_output(
        self, stream: StreamKey, input_epoch: CancellationEpoch, turn_id: str
    ) -> CancellationEpoch | None:
        """Allocate the output lease epoch for a finalized Brain reply.

        Mic's input epoch only validates incoming ASR.  It is deliberately
        independent from the monotonically increasing output lease generation.
        """
        output_fence = self._fence_for(stream)
        if output_fence is None:
            return input_epoch
        correlation = self._correlations.get(stream)
        if correlation is None:
            return None
        lease = output_fence.activate_for_turn(
            stream=stream,
            segment_id=SegmentId(
                f"onsite-response-{stream.stream_id}-{int(input_epoch)}"
            ),
            turn_id=turn_id,
        )
        if lease is None:
            return None
        callback = self._output_command_callback
        if callback is not None:
            await callback(stream, int(lease.cancellation_epoch))
        return lease.cancellation_epoch

    async def prepare_onsite_response_output(
        self, stream: StreamKey, input_epoch: CancellationEpoch, turn_id: str
    ) -> CancellationEpoch | None:
        """Admit a prepared first frame without interrupting current playback.

        A fresh stream can allocate immediately.  With an active lease, this
        is the sole route through the scheduler-owned Sound flush task; callers
        retain their first frame until its exact acknowledgement is committed.
        """
        output_fence = self._fence_for(stream)
        if output_fence is None:
            return input_epoch
        if not output_fence.has_active_lease(stream):
            return await self.allocate_onsite_response_output(
                stream, input_epoch, turn_id
            )
        correlation = self._correlations.get(stream)
        if (
            correlation is None
            or self._replacement_flush_callback is None
            or self._replacement_admit_callback is None
        ):
            return None
        prepared = output_fence.interrupt_for_turn(
            stream=stream,
            segment_id=SegmentId(f"agent-{turn_id}"),
            turn_id=turn_id,
            correlation=correlation,
        )
        if prepared is None:
            return None
        replacement, flush = prepared
        schedule = self._replacement_task_schedule
        task_id = schedule(stream, replacement, flush) if schedule is not None else None
        if task_id is False:
            _ = output_fence.abandon_replacement(stream)
            return None
        return await self._await_replacement_flush(stream, replacement, flush, task_id)

    @property
    def route_ready(self) -> bool:
        return any(stream in self._sinks for stream in self._mic_inputs)

    def register_control(
        self,
        raw_message: ControlEvent | str,
        peer_ip: str,
        owner: ConnectionId | None = None,
    ) -> None:
        parsed_event = (
            parse_control_event(raw_message)
            if isinstance(raw_message, str)
            else raw_message
        )

        match parsed_event:
            case MicInputRegistration(session_id=session_id, stream_id=stream_id):
                stream = StreamKey(session_id, stream_id)
                next_epoch = self._route_generations.get(stream, 0) + 1
                self._route_generations[stream] = next_epoch
                _ = self._last_asr_sequences.pop(stream, None)
                self._mic_inputs.add(stream)
                if owner is not None:
                    self._mic_input_owners[stream] = owner
                self._correlations[stream] = parsed_event.correlation
                observability = self._observability
                if observability is not None:
                    observability.bind_correlation(stream, parsed_event.correlation)

            case SinkRegistration(
                session_id=session_id, stream_id=stream_id, udp_port=udp_port
            ):
                self._register_sink(
                    StreamKey(session_id, stream_id), (peer_ip, udp_port), owner
                )

            case StreamState(
                session_id=session_id,
                stream_id=stream_id,
                state="cancelled" | "error",
            ):
                self._remove_stream(StreamKey(session_id, stream_id))

            case VoiceEvidence():
                callback = self._voice_evidence_callbacks.get(
                    parsed_event.session_id
                )
                if callback is not None:
                    _ = callback(parsed_event)

            case AsrFinal() | AsrPartial():
                # Finals are admitted by TransportRuntime, where they can enter
                # the session's common audience gate. Partials have no effects.
                return

            # Playback completion is not a disconnect.  Keeping the established
            # Mic/Sound route lets the next scheduler-authorized turn allocate a
            # fresh generated SSRC without requiring either peer to reconnect.
            case StreamReady() | StreamState() | StreamFlush() | FlushAcknowledgement():
                return

    def route_datagram(self, data: bytes, peer: PeerAddress) -> bool:
        """Reject ingress UDP: RTP is an Orchestrator-to-Sound output only.

        The listener remains attached because the same UDP transport emits
        generated TTS packets to Sound.  No Mic address, SSRC, or packet is
        accepted as an input route.
        """
        _ = data, peer
        return False

    async def deliver_generated_rtp(
        self, stream: StreamKey, epoch: CancellationEpoch, packet: bytes
    ) -> None:
        output_fence = self._fence_for(stream)

        if output_fence is None and epoch != CancellationEpoch(
            self._route_generations.get(stream, 0)
        ):
            self._log_rtp_rejection("route_epoch", stream, epoch, packet)
            return

        if output_fence is not None and not output_fence.can_emit(stream, epoch):
            self._log_rtp_rejection("output_fence", stream, epoch, packet)
            return

        if not _is_canonical_rtp(packet):
            self._log_rtp_rejection("invalid_packet", stream, epoch, packet)
            return

        sink = self._sinks.get(stream)

        transport = self._transport

        if sink is None or transport is None:
            self._log_rtp_rejection(
                "missing_sink" if sink is None else "missing_transport",
                stream,
                epoch,
                packet,
            )
            return

        transport.sendto(packet, sink)

        key = (stream, epoch)
        count = self._rtp_egress_counts.get(key, 0) + 1
        self._rtp_egress_counts[key] = count
        if count == 1:
            _LOGGER.debug(
                "rtp_egress_started session=%s stream=%s epoch=%d sink=%s:%d %s",
                stream.session_id,
                stream.stream_id,
                int(epoch),
                sink[0],
                sink[1],
                _rtp_packet_summary(packet),
            )

        self._record_rtp("rtp_egress", stream)

    def _log_rtp_rejection(
        self,
        reason: str,
        stream: StreamKey,
        epoch: CancellationEpoch,
        packet: bytes,
    ) -> None:
        key = (stream, epoch)
        if self._rtp_egress_counts.get(key, 0) != 0:
            return
        self._rtp_egress_counts[key] = -1
        _LOGGER.debug(
            "rtp_egress_rejected session=%s stream=%s epoch=%d reason=%s %s",
            stream.session_id,
            stream.stream_id,
            int(epoch),
            reason,
            _rtp_packet_summary(packet),
        )

    def _record_rtp(self, stage: OnsiteStage, stream: StreamKey) -> None:
        observability = self._observability

        if observability is not None:
            observability.record_stream(stage, stream)

    async def wait_for_onsite_jobs(self) -> None:
        bridge = self._onsite_bridge

        if bridge is not None:
            await bridge.wait_quiescent()

    def remove_connection(self, owner: ConnectionId) -> None:
        for stream, input_owner in tuple(self._mic_input_owners.items()):
            if input_owner == owner:
                self._invalidate_stream(stream)
                self._mic_inputs.discard(stream)
                _ = self._mic_input_owners.pop(stream, None)
                _ = self._last_asr_sequences.pop(stream, None)
        for stream, route_owner in tuple(self._sink_owners.items()):
            if route_owner == owner:
                self._remove_sink(stream)

    def clear(self) -> None:
        self._mic_inputs.clear()

        self._correlations.clear()

        self._last_asr_sequences.clear()

        self._sinks.clear()

        self._sink_owners.clear()

        self._voice_evidence_callbacks.clear()
        self._output_fences.clear()

    def remove_stream(self, session_id: str, stream_id: str) -> None:
        self._remove_stream(StreamKey(session_id, stream_id))

    def remove_session(self, session_id: str) -> None:
        _ = self._voice_evidence_callbacks.pop(session_id, None)
        _ = self._output_fences.pop(session_id, None)
        streams = {
            *(stream for stream in self._mic_inputs if stream.session_id == session_id),
            *(stream for stream in self._sinks if stream.session_id == session_id),
            *(
                stream
                for stream in self._correlations
                if stream.session_id == session_id
            ),
        }
        for stream in streams:
            self._remove_stream(stream)

    def output_ssrc(self, stream: StreamKey, cancellation_epoch: int = 0) -> int:
        return generated_ssrc(stream, CancellationEpoch(cancellation_epoch))

    def correlation(self, stream: StreamKey) -> EnvelopeCorrelation | None:
        return self._correlations.get(stream)

    def input_epoch(self, stream: StreamKey) -> int:
        return self._route_generations.get(stream, 0)

    def owns_mic_input(self, stream: StreamKey, owner: ConnectionId) -> bool:
        return self._mic_input_owners.get(stream) == owner

    def _register_sink(
        self,
        stream: StreamKey,
        endpoint: PeerAddress,
        owner: ConnectionId | None,
    ) -> None:
        if stream in self._sinks:
            raise DuplicateRouteError(stream)

        self._sinks[stream] = endpoint

        if owner is not None:
            self._sink_owners[stream] = owner

    def _remove_stream(self, stream: StreamKey) -> None:
        self._remove_sink(stream)

    def _remove_sink(self, stream: StreamKey) -> None:
        self._invalidate_stream(stream)

        _ = self._sinks.pop(stream, None)

        _ = self._sink_owners.pop(stream, None)

    def _invalidate_stream(self, stream: StreamKey) -> None:
        next_generation = self._route_generations.get(stream, 0) + 1

        self._route_generations[stream] = next_generation

        if self._onsite_bridge is not None:
            self._onsite_bridge.invalidate_stream(
                stream, CancellationEpoch(next_generation)
            )

def _is_canonical_rtp(data: bytes) -> bool:
    return (
        len(data) == RTP_HEADER_BYTES + L16_FRAME_BYTES
        and data[0] == RTP_V2_HEADER
        and data[1] & 0x7F == RTP_PAYLOAD_TYPE
    )


def _rtp_packet_summary(packet: bytes) -> str:
    has_header = len(packet) >= RTP_HEADER_BYTES
    sequence = int.from_bytes(packet[2:4], "big") if has_header else -1
    ssrc = int.from_bytes(packet[8:12], "big") if has_header else -1
    digest = hashlib.sha256(packet).hexdigest()[:16]
    return f"bytes={len(packet)} sequence={sequence} ssrc={ssrc} sha256={digest}"
