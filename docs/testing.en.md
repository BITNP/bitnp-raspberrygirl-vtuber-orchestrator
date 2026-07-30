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

Use `bash scripts/verify_workspace.sh --sibling-root ..` only when every sibling path is present. This Contract gate runs the schema, topology, and Frontend contract verifiers; it does not run Local tests or optional release gates. Real adapter smoke tests are explicit opt in with `BITNP_REAL_ADAPTER_FAKE_LOCAL=1`.

## Onsite Acceptance Matrix

Normal CI stays credential-free and hardware-free. It runs the deterministic local
provider chain, deployment artifacts, and generated Sound RTP acceptance:

```bash
uv run pytest tests/integration/test_streaming_onsite_acceptance.py tests/test_deployment_topology.py -q
python scripts/verify_topology.py --deployment-root ..
```

Provider compatibility is opt in. This uses the in-process fake-local adapter
server and does not need a credential, GPU, microphone, or speaker:

```bash
BITNP_REAL_ADAPTER_FAKE_LOCAL=1 uv run pytest -m real_adapter -q
```

LAN WSS/TLS and physical 16 kHz device acceptance are manual, opt-in deployment
checks. They fail closed at service startup when the required WSS URL, TLS
material, trusted-LAN token, or provider configuration is absent. Start the
deployed commands in this exact order, then retain the `queued` and `playing`
Sound control envelopes with the deployment record:

```bash
uv run orchestrator-transport
uv run sound-receive
uv run mic-stream
```

For the device loop, set the protected production environment, including
`TRUSTED_LAN_TOKEN`, WSS/TLS paths, provider settings, `BITNP_CAPTURE_DEVICE`,
and `BITNP_PLAYBACK_DEVICE`; do not run this command in normal CI. The evidence
must show the shared session and stream IDs plus generated 16 kHz L16 RTP, never
the token, certificates, raw Mic media, or device identifiers.
