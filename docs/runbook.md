# Orchestrator Runbook

Run the local deterministic gate from this repository:

```bash
uv sync --locked
uv run pytest
uv run basedpyright
uv run ruff check src tests
python scripts/verify_protocol_schema.py
python scripts/verify_topology.py --sibling-root ..
python scripts/verify_vtuber_contract.py --frontend-path ../bitnp-raspberrygirl-vtuber-frontend
```

`scripts/verify_workspace.sh` is the optional explicit-sibling integration gate. It writes no evidence files and does not rely on a parent Git checkout. `ORCHESTRATOR_LLM_PROVIDER=mock` is the normal credential-free default.

## Operational Journal Boundary

Onsite transport creates one shared in-process recorder before listeners start. Its
release-safe journal retains only hashed correlation identifiers and bounded stage
or outcome labels. It is an incident/export boundary, not session state: do not
export raw audio, templates, credentials, prompts, or tool payloads, and retain
only the sanitized evidence artifact required by the deployment record.

Run the lecturer demo with `python scripts/run_lecturer_demo.py --script samples/lecturer/bitnet_intro_zh.json --evidence .omo/evidence/lecturer-demo.json`. It emits `media.stream.command`, `media.stream.state`, RTP-timed cues, and explicit positive slide-page commands.
