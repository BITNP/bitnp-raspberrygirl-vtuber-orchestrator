# Architecture

Orchestrator is the hub for Mic, Comments, Sound, and the frontend. It alone makes cross service routing and provider decisions. Mic and Sound use RTP with the hub; all control traffic uses the canonical protocol.

Only Orchestrator and the frontend are mode aware. They interpret `lecturer`, `virtual_streamer`, and `onsite_explainer`. Orchestrator owns configurable OpenAI compatible ASR, LLM, and TTS providers, while the other services remain mode agnostic.
