# Onsite Spoken-Dialogue Deployment

These templates run the real onsite loop:

```text
Mic VAD/ASR -> Orchestrator Gate/LLM -> vLLM-Omni TTS -> Sound
```

The Orchestrator is the only hub. Mic and Sound each connect only to its `/control` WSS endpoint and its RTP listener. They remain strategy-agnostic. Comments can submit audience input to the same WSS endpoint, but is outside the onsite audio loop. Frontend is not part of this deployment.

## Prepare Hosts

1. Install each repository and run `uv sync --locked` on its host. The unit templates assume `/opt/bitnp/bitnp-raspberrygirl-vtuber-{orchestrator,mic,sound}` and the `bitnp` user and group. Replace paths, user, group, and Python virtual-environment executable paths for the target system.
2. Copy each service's `.env.example` to the matching `/etc/bitnp/*.env` file. `EnvironmentFile` injects values directly. There is no dotenv loader. For a non-systemd local command, `uv run --env-file .env <command>` is the explicit equivalent; plain `uv run <command>` does not load `.env`.
3. Keep secrets outside the repositories. Inject `TRUSTED_LAN_TOKEN`, the LLM API key, and the optional Mic ASR API key through protected environment files or the site's secret mechanism. Mount the Orchestrator TLS certificate, TLS private key, voice-reference WAV, and one PEM CA bundle read-only at the paths named in the environment files. Set `ORCHESTRATOR_TLS_CA_PATH` to that bundle in Orchestrator, Mic, Sound, and Comments. The bundle may contain the internal root and any intermediates. Orchestrator uses it for self-hosted LLM and TTS HTTPS endpoints, while Mic, Sound, and Comments use it to verify the Orchestrator WSS certificate. Protect these files so only the service account can read them. Host system trust remains an optional alternative when it already contains the issuing CA, but it isn't the deployment contract.
4. Set `ORCHESTRATOR_LLM_PROVIDER=openai_compatible`, explicitly select `ORCHESTRATOR_LLM_REASONING_DIALECT=deepseek` or `openai`, and set `ORCHESTRATOR_TTS_PROVIDER=vllm_omni`. Configure `MIC_ASR_ENDPOINT` and `MIC_ASR_MODEL` in Mic's environment. Supply every LLM/TTS provider endpoint and model variable in the Orchestrator example, the LLM API key, and nonempty `ORCHESTRATOR_TTS_VOICE`, `ORCHESTRATOR_TTS_REF_AUDIO`, and `ORCHESTRATOR_TTS_REF_TEXT`. Optional Gate, Brain, and Maintenance model variables fall back to `ORCHESTRATOR_LLM_MODEL`. `ORCHESTRATOR_TTS_VOICE` must be an ID accepted by the configured vLLM-Omni server. `ORCHESTRATOR_TTS_REF_AUDIO` may be an absolute local WAV path, a `file://` WAV URI, or an existing `data:audio/wav;base64,...` URL. Orchestrator encodes local paths and file URIs into the TTS request so the provider does not need host filesystem access; do not put a real data URL with private voice material in source control.
5. Configure Mic and Sound with the same `BITNP_SESSION_ID` or `SOUND_SESSION_ID` and the same `BITNP_MIC_RTP_STREAM_ID` or `SOUND_RTP_STREAM_ID`. Configure Mic, Sound, and Comments with `wss://<orchestrator-host>/control`, the trusted-LAN token, and the shared CA-bundle path. Mic also needs the Orchestrator RTP host and port. No Mic-to-Sound address belongs in either file.
6. On the private LAN, allow TCP to the Orchestrator control port and UDP to the Orchestrator RTP port from Mic and Sound. Allow UDP to the Sound RTP port from Orchestrator. Keep all of these listeners off public networks. If advertised and bound ports differ, create the matching forwarding rules.
7. Install PortAudio on Mic and Sound hosts. Grant the service account access to the selected recording and playback devices under the host's audio policy. Select and verify devices with `BITNP_CAPTURE_DEVICE` and `BITNP_PLAYBACK_DEVICE`; do not rely on a desktop default without testing it, because a monitor/loopback source or a noisy source can manufacture false speech endpoints. On a PipeWire/PulseAudio desktop, a system unit running as `bitnp` does not automatically inherit the logged-in user's audio session. Run it in a service account with a configured audio session, or add a host-specific systemd drop-in for the correct audio runtime environment after verifying device access. An `audio` group alone is not a complete PipeWire deployment configuration.
8. Install the three units in `/etc/systemd/system/`, then run:

   ```bash
   systemctl daemon-reload
   systemctl enable orchestrator-transport.service sound-receive.service mic-stream.service
   ```

   The ordering directives apply when all services share a host. For separate hosts, start Orchestrator first, wait for the WSS listener to accept the authenticated registrations, then start Sound and Mic in that order.

The templates contain no credentials, certificate data, or private keys. They use `Restart=on-failure`; inspect `journalctl -u <unit>` after any restart rather than treating process start as an acceptance check.

Before copying protected environment files to hosts, validate the checked-in unit
and sanitized environment shape from the Orchestrator checkout:

```bash
python scripts/verify_topology.py --deployment-root ..
```

This rejects missing PEM CA-bundle paths, non-Orchestrator control endpoints, Mic/Sound direct-peer endpoints, and unequal `BITNP_SESSION_ID` / `SOUND_SESSION_ID` or `BITNP_MIC_RTP_STREAM_ID` / `SOUND_RTP_STREAM_ID` values.

## Single-Host Model Servers

Mic ASR and Orchestrator TTS model servers on the same host may use loopback HTTP endpoints such as `http://127.0.0.1:8090/v1/audio/transcriptions` and `http://127.0.0.1:8091/v1`. This exception applies only to the private, same-host provider boundary. The Mic/Sound/Comments control plane must still use authenticated WSS with the CA bundle and `TRUSTED_LAN_TOKEN`; do not enable insecure loopback WebSocket mode for this deployment.

For `EnvironmentFile`, keep `ORCHESTRATOR_TTS_REF_TEXT` to one properly quoted line. If the reference transcript is multiline, provide an escaped single-line value or use the site's secret/environment injection mechanism rather than pasting unescaped newlines into the file.

## Start, Roll Back, And Accept

1. Start `orchestrator-transport.service`, then `sound-receive.service`, then `mic-stream.service`. On a shared host, start all three units in one command so systemd applies the declared ordering:

   ```bash
   systemctl start orchestrator-transport.service sound-receive.service mic-stream.service
   ```
2. Confirm the live WSS registrations use the expected shared session and stream identities. Sound must receive the Orchestrator command and emit its `media.rtp.sink.ready` event before Mic sends the acceptance utterance. The runtime has no HTTP health endpoint, so don't substitute an HTTP probe for this control-plane check.
3. Speak a short question with a deliberately recognizable voice and wording into Mic for at least one second, then remain quiet until playback starts. Mic performs VAD/endpointing on 20 ms PCM16 frames, sends the endpointed 16 kHz mono PCM WAV only to its configured ASR provider, then emits one authenticated `asr.final`. Orchestrator validates its route and epoch, runs the shared Gate and Brain, requests TTS, converts accepted mono PCM16 TTS WAV into canonical 16 kHz audio when needed, then sends generated 16 kHz L16 RTP to Sound.
4. Observe Sound's control session report `media.stream.state` `queued`, `playing`, and finally `finished`. Hear Sound play the generated answer in the configured TTS voice. It must not replay the spoken source audio. The `finished` event must correlate to the current turn, segment, epoch, and SSRC before the next output lease is released.
5. Repeat the test for a second natural turn. During a third answer, speak a new valid question to verify barge-in: the old output must stop, Sound must acknowledge the epoch-correlated flush, and no old RTP may resume before the replacement command starts.
6. For latency or audio diagnostics, inspect Mic's endpoint/ASR logs, Orchestrator's Gate/Brain/TTS records, and Sound's `rtp_ingress_diagnostic`. A healthy generated stream has `drops=0` and `late_gaps=0`; use these records to distinguish endpointing, ASR, Gate/LLM, TTS, RTP, and playback delay.
7. For a normal shutdown or a failed test, stop Mic, then Sound, then Orchestrator. Restore the prior protected environment files and start in the documented order. Do not change schemas, open a direct Mic-to-Sound path, or use loopback WS for a LAN rollback.

Local, single-host loopback development is documented separately in the [local loopback guide](../docs/local-loopback.zh-CN.md). It is not a production deployment option.

系统架构、协议和验证命令见[Orchestrator 开发者文档](../docs/developer.zh-CN.md)。
