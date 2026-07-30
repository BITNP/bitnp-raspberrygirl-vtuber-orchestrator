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

## Onsite Spoken-Dialogue Loop

The real onsite loop is enabled by the provider combination `ORCHESTRATOR_ASR_PROVIDER=openai_compatible`, `ORCHESTRATOR_LLM_PROVIDER=openai_compatible`, and `ORCHESTRATOR_TTS_PROVIDER=vllm_omni`, plus every matching endpoint and model variable in `.env.example`. Supply the provider API keys through secret injection. The LLM API key is required. Set nonempty `ORCHESTRATOR_TTS_VOICE`, `ORCHESTRATOR_TTS_REF_AUDIO`, and `ORCHESTRATOR_TTS_REF_TEXT`; the voice-reference WAV path must be mounted read-only outside source control.

For each accepted Mic utterance, the bridge applies the documented deterministic VAD endpoint before sending fixed 16 kHz mono PCM WAV to ASR: threshold 400, ten-frame pre-roll, 600 ms silence close, and 15 s forced close. It then obtains an LLM answer, sends it to the vLLM-Omni TTS extension, validates its fixed 16 kHz mono PCM WAV response, and emits generated L16 RTP to Sound. It does not forward raw Mic RTP in this path.

Mic and Sound must use the same session ID and stream ID, the deployed `wss://<host>/control` endpoint, and the shared trusted-LAN token. They remain strategy-agnostic and have no direct peer endpoint. Frontend is excluded from this onsite audio deployment. See [the systemd deployment bundle](../deploy/README.md) for secret mounts, private-LAN firewall rules, PortAudio access, startup and rollback ordering, and the live generated-TTS acceptance test.

Validate the checked-in systemd and sanitized Mic/Sound environment manifests before
deployment:

```bash
python scripts/verify_topology.py --deployment-root ..
```

For normal, credential-free provider and deployment acceptance, run:

```bash
uv run pytest tests/integration/test_streaming_onsite_acceptance.py tests/test_deployment_topology.py -q
```

The fake-local provider compatibility smoke is explicitly opt in:

```bash
BITNP_REAL_ADAPTER_FAKE_LOCAL=1 uv run pytest -m real_adapter -q
```

Production WSS/TLS and 16 kHz capture/playback checks need the protected provider,
token, certificate, and device environment. Missing values reject startup; keep the
sanitized `queued` and `playing` Sound envelopes as deployment acceptance evidence.

## Measured Streaming Rollout

Before promoting streaming ASR, diarization, or consented recognition, run:

```bash
uv run python scripts/benchmark_multimodal.py --fixtures tests/fixtures/multimodal_benchmark --baseline .omo/evidence/asr-baseline.json --report .omo/evidence/task-8-benchmark.json --max-cer-regression-pp 1.0 --max-duplicate-turns 0 --require-p95-final-latency-improvement-percent 20
uv run python scripts/verify_plan_contracts.py --plan .omo/plans/multimodal-agent-scheduler.md --require-chinese-prompts --forbid-raw-mic-to-sound --require-task-snapshot-validation
uv run python scripts/verify_scheduler_scope.py --forbid-peer-links --forbid-biometric-authorization --require-closed-command-validation --require-memory-provenance
```

The report records provider/model/configuration/corpus versions, CER, p95 final latency, stale and duplicate turns, and aggregate shadow-memory decisions with provenance completeness. Fixtures and reports contain synthetic text and opaque IDs only: never add audio, recordings, memory values, raw prompts, credentials, or biometric templates.

Threshold failures block promotion. To roll back, restore the recorded final-only ASR provider/model/configuration values in the protected environment, stop Mic then Sound then Orchestrator, and restart Orchestrator then Sound then Mic. Re-run the same benchmark corpus and retain the sanitized before/after reports. Never replace this rollback with raw Mic RTP forwarding or a direct Mic-to-Sound path.
