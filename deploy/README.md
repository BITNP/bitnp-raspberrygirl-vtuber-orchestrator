# Onsite Spoken-Dialogue Deployment

These templates run the real onsite loop:

```text
Mic -> Orchestrator ASR -> Orchestrator LLM -> vLLM-Omni TTS -> Sound
```

The Orchestrator is the only hub. Mic and Sound each connect only to its `/control` WSS endpoint and its RTP listener. They remain strategy-agnostic. Comments can submit audience input to the same WSS endpoint, but is outside the onsite audio loop. Frontend is not part of this deployment.

## Prepare Hosts

1. Install each repository and run `uv sync --locked` on its host. The unit templates assume `/opt/bitnp/bitnp-raspberrygirl-vtuber-{orchestrator,mic,sound}` and the `bitnp` user and group. Replace paths, user, group, and Python virtual-environment executable paths for the target system.
2. Copy each service's `.env.example` to the matching `/etc/bitnp/*.env` file. `EnvironmentFile` injects values directly. There is no dotenv loader.
3. Keep secrets outside the repositories. Inject `TRUSTED_LAN_TOKEN` and provider API keys through the protected environment files or the site's secret mechanism. Mount the Orchestrator TLS certificate, TLS private key, voice-reference WAV, and one PEM CA bundle read-only at the paths named in the environment files. Set `ORCHESTRATOR_TLS_CA_PATH` to that bundle in Orchestrator, Mic, Sound, and Comments. The bundle may contain the internal root and any intermediates. Orchestrator uses it for self-hosted ASR, LLM, and TTS HTTPS endpoints, while Mic, Sound, and Comments use it to verify the Orchestrator WSS certificate. Protect these files so only the service account can read them. Host system trust remains an optional alternative when it already contains the issuing CA, but it isn't the deployment contract.
4. Set `ORCHESTRATOR_LLM_PROVIDER=openai_compatible`, `ORCHESTRATOR_ASR_PROVIDER=openai_compatible`, and `ORCHESTRATOR_TTS_PROVIDER=vllm_omni`. Supply every self-hosted provider endpoint and model variable in the Orchestrator example, the LLM API key, and nonempty `ORCHESTRATOR_TTS_VOICE`, `ORCHESTRATOR_TTS_REF_AUDIO`, and `ORCHESTRATOR_TTS_REF_TEXT`.
5. Configure Mic and Sound with the same `BITNP_SESSION_ID` or `SOUND_SESSION_ID` and the same `BITNP_MIC_RTP_STREAM_ID` or `SOUND_RTP_STREAM_ID`. Configure Mic, Sound, and Comments with `wss://<orchestrator-host>/control`, the trusted-LAN token, and the shared CA-bundle path. Mic also needs the Orchestrator RTP host and port. No Mic-to-Sound address belongs in either file.
6. On the private LAN, allow TCP to the Orchestrator control port and UDP to the Orchestrator RTP port from Mic and Sound. Allow UDP to the Sound RTP port from Orchestrator. Keep all of these listeners off public networks. If advertised and bound ports differ, create the matching forwarding rules.
7. Install PortAudio on Mic and Sound hosts. Grant the service account access to the selected recording and playback devices under the host's audio policy. Select devices with `BITNP_CAPTURE_DEVICE` and `BITNP_PLAYBACK_DEVICE` when the defaults are unsuitable.
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

## Start, Roll Back, And Accept

1. Start `orchestrator-transport.service`, then `sound-receive.service`, then `mic-stream.service`. On a shared host, start all three units in one command so systemd applies the declared ordering:

   ```bash
   systemctl start orchestrator-transport.service sound-receive.service mic-stream.service
   ```
2. Confirm the live WSS registrations use the expected shared session and stream identities. Sound must receive the Orchestrator command and emit its `media.rtp.sink.ready` event before Mic sends the acceptance utterance. The runtime has no HTTP health endpoint, so don't substitute an HTTP probe for this control-plane check.
3. Speak a short question with a deliberately recognizable voice and wording into Mic for at least one second. The bridge batches fifty 20 ms Mic frames, sends fixed 16 kHz mono PCM WAV to ASR, obtains an LLM answer, requests TTS, validates the returned 16 kHz mono WAV, then sends generated L16 RTP to Sound.
4. Observe Sound's control session report `media.stream.state` `queued` and then `playing`. Hear Sound play the generated answer in the configured TTS voice. It must not replay the spoken source audio. That audible difference, together with the queued and playing events, is the acceptance evidence that generated TTS, not raw Mic RTP, reached Sound.
5. If the test fails, stop Mic, then Sound, then Orchestrator. Restore the prior protected environment files and start in the same order. Do not change schemas, open a direct Mic-to-Sound path, or use loopback WS for a LAN rollback.

系统架构、协议和验证命令见[Orchestrator 开发者文档](../docs/developer.zh-CN.md)。
