# Architecture

Orchestrator is the hub for Mic, Comments, Sound, and the frontend. It alone makes cross service routing and provider decisions. Mic and Sound use RTP with the hub; all control traffic uses the canonical protocol.

Only Orchestrator and the frontend are mode aware. They interpret `lecturer`, `virtual_streamer`, and `onsite_explainer`. Orchestrator owns configurable OpenAI compatible ASR, LLM, and TTS providers, while the other services remain mode agnostic.

## Onsite Explainer Audio Bridge

`onsite_explainer` is a hub-owned audio replacement path. `transport_app.py` enables its bridge only when this mode is active. Mic and Sound still use their normal WSS control exchange with Orchestrator and their normal RTP boundary, so neither service needs mode-specific behavior. The frontend is excluded from this audio deployment.

```text
Mic  -- WSS source control --> Orchestrator <-- WSS sink control -- Sound
Mic  -- L16 RTP over UDP --> Orchestrator -- generated L16 RTP over UDP --> Sound
                              |
+-> VAD endpoint -> ASR -> turn pipeline -> LLM -> TTS
```

WSS control registers the Mic source and Sound sink for the same session and stream. Orchestrator validates and pins the accepted Mic RTP route, then sends Sound a `media.stream.command` with the generated output SSRC. In onsite mode, raw Mic RTP is not forwarded. Generated audio replaces it at the Sound RTP boundary.

The bridge endpoints accepted 20 ms L16 RTP frames with deterministic VAD: a sample magnitude of at least 400 starts speech, up to ten 20 ms pre-roll frames are retained, thirty silent frames (600 ms) close an utterance, and 750 frames (15 s) force a close. Duplicates drop, one-frame reordering is accepted, gaps close the current utterance, and disconnect closes it once. At the provider boundary, network-order L16 samples are byte-swapped into PCM16LE and wrapped as a 16 kHz, mono PCM WAV for ASR. The configured OpenAI-compatible ASR, OpenAI-compatible LLM, and vLLM-Omni TTS compose through the onsite turn pipeline, and run off the UDP callback in worker-thread offload. TTS output must be `audio/wav`, 16 kHz, mono, uncompressed PCM16. The bridge byte-swaps validated PCM16LE WAV samples back to network-order L16, pads them to whole RTP payloads, and packetizes them with a deterministic nonzero generated SSRC derived from the Mic SSRC.

Provider or media failure produces no generated output. A blank ASR result also produces no output. Stream cancellation, removal, disconnect, and runtime clearing invalidate pending bridge jobs. Route generations gate completion, so a cancelled or disconnected route drops late output rather than sending it to Sound.

Tests cover the L16 and PCM16LE byte-order conversions, the fixed WAV validation, blank-ASR no-output behavior, Sound command and generated-packet compatibility, and cancellation suppression of stale output. See the [deployment documentation](deployment.en.md) for configuration and operational procedures.
