# Quickstart

Use Python 3.12 or later. The default `ORCHESTRATOR_LLM_PROVIDER=mock` path needs no credentials, GPU, external service, RTP hardware, or Godot.

```bash
uv sync --locked
uv run pytest
uv run basedpyright
uv run ruff check src tests
```

For local contract checks, run `python scripts/verify_protocol_schema.py` and `python scripts/verify_topology.py --sibling-root ..`.
