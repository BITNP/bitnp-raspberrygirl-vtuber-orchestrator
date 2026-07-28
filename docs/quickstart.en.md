# Quickstart

Use Python 3.12 or later. The default `ORCHESTRATOR_LLM_PROVIDER=mock` path needs no credentials, GPU, external service, RTP hardware, or Godot.

```bash
uv sync --locked
uv run pytest
uv run basedpyright
uv run ruff check src tests
```

For local contract checks, run `python scripts/verify_protocol_schema.py` and `python scripts/verify_topology.py --sibling-root ..`.

For a real onsite explainer deployment, use `ORCHESTRATOR_MODE=onsite_explainer` with the OpenAI-compatible ASR and LLM settings and the `vllm_omni` TTS settings in `.env.example`. The runtime command is `uv run orchestrator-transport`. Start it before Sound and Mic. See the [deployment guide](deployment.en.md) and [systemd bundle](../deploy/README.md); frontend is not part of this loop.
