# Testing

Run the local checks from this checkout:

```bash
uv run pytest
uv run basedpyright
uv run ruff check src tests
python scripts/verify_protocol_schema.py
python scripts/verify_topology.py --sibling-root ..
python scripts/verify_vtuber_contract.py --frontend-path ../bitnp-raspberrygirl-vtuber-frontend
```

Use `bash scripts/verify_workspace.sh` only when every sibling path is present. Real adapter smoke tests are explicit opt in with `BITNP_REAL_ADAPTER_FAKE_LOCAL=1`.
