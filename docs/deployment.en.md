# Deployment

Keep `ORCHESTRATOR_LLM_PROVIDER=mock` for credential free development. For OpenAI compatible ASR, LLM, or TTS, set the matching `ORCHESTRATOR_ASR_*`, `ORCHESTRATOR_LLM_*`, and `ORCHESTRATOR_TTS_*` endpoint, model, and API key placeholders in the deployment environment. vLLM Omni voice cloning is an OpenAI compatible TTS extension. Never commit real keys.

## Transport Runtime

Run `orchestrator-transport` as the Orchestrator owned hub process. It listens for authenticated WSS control and UDP RTP. Mic and Sound connect to this process only. Do not create service to service peer links, and do not copy the canonical schemas out of this repository.

The deployment environment must set every transport variable in `.env.example`:

- `ORCHESTRATOR_CONTROL_BIND_HOST` and `ORCHESTRATOR_CONTROL_BIND_PORT` select the local WSS listener.
- `ORCHESTRATOR_RTP_BIND_HOST` and `ORCHESTRATOR_RTP_BIND_PORT` select the local UDP RTP listener.
- `ORCHESTRATOR_TRANSPORT_ADVERTISED_HOST`, `ORCHESTRATOR_TRANSPORT_ADVERTISED_CONTROL_PORT`, and `ORCHESTRATOR_TRANSPORT_ADVERTISED_RTP_PORT` identify the reachable LAN endpoint for peer configuration. Set them to the address and ports that Mic and Sound can reach. They may differ from the bind settings only when the matching forwarding rule exists.
- `TRUSTED_LAN_TOKEN` is the shared bearer token for WSS handshakes. Store its real value in the deployment secret store, not in `.env.example`, documentation, source control, or command history.
- `ORCHESTRATOR_CONTROL_TLS_CERT_PATH` and `ORCHESTRATOR_CONTROL_TLS_KEY_PATH` are read only paths to a certificate and private key provisioned outside this repository. The runtime loads this pair itself. Do not generate, embed, or commit TLS material here.

Production always uses WSS. The process rejects a nonloopback configuration without both TLS paths and `TRUSTED_LAN_TOKEN`. If an external port forwards to the control bind port, it must preserve the WSS TLS connection because the runtime owns the certificate and key. Limit the control and RTP listeners to the private LAN. Allow TCP to the reachable control port and UDP to the reachable RTP port only from the Mic and Sound networks. Do not expose either listener publicly or rely on the bearer token as a substitute for firewall policy.

`ORCHESTRATOR_TRANSPORT_ALLOW_LOOPBACK_WS=true` is an explicit test exception. It permits plain WS without TLS or a token only when the control bind host, RTP bind host, and advertised host are all `127.0.0.1`, `::1`, or `localhost`. It is not a production or LAN diagnostic mode.

## Route And Listener Readiness

The runtime is listener ready only after both the UDP RTP listener and WSS control listener have started. It has no stdin or stdout control protocol and no HTTP readiness endpoint. Supervise the long running process with the service manager and treat a failed startup as not ready.

An RTP route needs an authenticated source registration and sink registration for the same session and stream. The first valid source RTP packet pins the source UDP port. Forwarding starts only when that pinned source and a sink route exist. Canonical source and sink ready messages are accepted, but they do not by themselves prove that RTP can flow. The hub forwards only canonical RTP V2, payload type 96, L16 packets and drops other UDP input.

## Start And Verify

After the deployment system has supplied the environment and mounted the externally provisioned TLS files, start the process with:

```bash
uv run orchestrator-transport
```

Run these checks from the Orchestrator repository before deployment:

```bash
uv sync --locked
uv run pytest tests/test_transport_contract.py tests/test_transport_runtime.py tests/integration/test_rtp_transport_loopback.py
uv run basedpyright
uv run ruff check src tests
python scripts/verify_protocol_schema.py
python scripts/verify_topology.py --sibling-root ..
python scripts/verify_vtuber_contract.py --frontend-path ../bitnp-raspberrygirl-vtuber-frontend
bash scripts/verify_workspace.sh --sibling-root ..
```

The loopback integration check verifies the WS and UDP test path only. It does not replace a private LAN WSS deployment with externally provisioned TLS files and firewall rules.

## Onsite Explainer Loop

Set `ORCHESTRATOR_MODE=onsite_explainer` only for the real onsite loop. It requires `ORCHESTRATOR_ASR_PROVIDER=openai_compatible`, `ORCHESTRATOR_LLM_PROVIDER=openai_compatible`, and `ORCHESTRATOR_TTS_PROVIDER=vllm_omni`, plus every matching endpoint and model variable in `.env.example`. Supply the provider API keys through secret injection. The LLM API key is required. Set nonempty `ORCHESTRATOR_TTS_VOICE`, `ORCHESTRATOR_TTS_REF_AUDIO`, and `ORCHESTRATOR_TTS_REF_TEXT`; the voice-reference WAV path must be mounted read-only outside source control.

For each accepted Mic utterance, the bridge batches fifty 20 ms canonical L16 frames, sends a fixed 16 kHz mono PCM WAV to ASR, obtains an LLM answer, sends it to the vLLM-Omni TTS extension, validates its fixed 16 kHz mono PCM WAV response, and emits generated L16 RTP to Sound. It does not forward raw Mic RTP in this mode.

Mic and Sound must use the same session ID and stream ID, the deployed `wss://<host>/control` endpoint, and the shared trusted-LAN token. They remain mode-agnostic and have no direct peer endpoint. Frontend is excluded from this onsite audio deployment. See [the systemd deployment bundle](../deploy/README.md) for secret mounts, private-LAN firewall rules, PortAudio access, startup and rollback ordering, and the live generated-TTS acceptance test.
