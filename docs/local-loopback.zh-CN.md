# 本机回环联调指南

本指南用于同一台开发机上的 Mic、Orchestrator 和 Sound 联调。它使用 `ws://` 和 loopback UDP，不能替代现场/LAN 部署，不能绑定或通告非 loopback 地址，也不能与生产 WSS 配置混用。

## 前提

- 在三个服务仓库分别完成 `uv sync --locked`。
- 每个服务都有自己的本地 `.env`；它不应提交。进程不会自动读取该文件，必须用 `uv run --env-file .env <command>` 启动。
- Mic 需要可用的 OpenAI-compatible ASR provider；Orchestrator 需要可用的 LLM、vLLM-Omni TTS provider。特别是 `ORCHESTRATOR_TTS_VOICE`、`ORCHESTRATOR_TTS_REF_AUDIO` 和 `ORCHESTRATOR_TTS_REF_TEXT` 都必须非空，且 voice ID 必须被 TTS 服务接受。
- Mic 与 Sound 使用完全相同的 session ID 与 stream ID。

## 配置

从各仓库的 `.env.example` 复制出本地 `.env`，保留各 provider 的实际 endpoint、model 和语音参考配置，再覆盖以下 transport 项。

Orchestrator `.env`：

```dotenv
ORCHESTRATOR_CONTROL_BIND_HOST=127.0.0.1
ORCHESTRATOR_CONTROL_BIND_PORT=8443
ORCHESTRATOR_RTP_BIND_HOST=127.0.0.1
ORCHESTRATOR_RTP_BIND_PORT=5004
ORCHESTRATOR_TRANSPORT_ADVERTISED_HOST=127.0.0.1
ORCHESTRATOR_TRANSPORT_ADVERTISED_CONTROL_PORT=8443
ORCHESTRATOR_TRANSPORT_ADVERTISED_RTP_PORT=5004
ORCHESTRATOR_TRANSPORT_ALLOW_LOOPBACK_WS=true
TRUSTED_LAN_TOKEN=
ORCHESTRATOR_CONTROL_TLS_CERT_PATH=
ORCHESTRATOR_CONTROL_TLS_KEY_PATH=
```

Mic `.env`：

```dotenv
ORCHESTRATOR_WS_URL=ws://localhost:8443/control
MIC_ALLOW_LOOPBACK_WS=true
TRUSTED_LAN_TOKEN=
BITNP_SESSION_ID=session-onsite-001
BITNP_MIC_STREAM_ID=onsite-primary
MIC_ASR_ENDPOINT=http://127.0.0.1:8090/v1/audio/transcriptions
MIC_ASR_MODEL=<openai-compatible-asr-model>
```

Sound `.env`：

```dotenv
ORCHESTRATOR_WS_URL=ws://localhost:8443/control
SOUND_ALLOW_LOOPBACK_WS=true
TRUSTED_LAN_TOKEN=
SOUND_RTP_BIND_HOST=127.0.0.1
SOUND_RTP_BIND_PORT=5006
SOUND_RTP_ADVERTISED_HOST=127.0.0.1
SOUND_SESSION_ID=session-onsite-001
SOUND_RTP_STREAM_ID=onsite-primary
```

回环 Orchestrator 不使用 bearer token，并会拒绝携带 `Authorization` header 的连接。因此 Mic 和 Sound 的 `TRUSTED_LAN_TOKEN` 必须为空；令牌相同也不能用于这个模式。回到 WSS/LAN 配置时，关闭两个 `*_ALLOW_LOOPBACK_WS` 开关，恢复 CA 与 TLS 配置，并让 Mic、Sound 分别使用与 Orchestrator 对应角色配置匹配、彼此不同的非空令牌。

## 启动与验证

在三个不同终端、各自仓库中按以下顺序运行：

```bash
uv run --env-file .env orchestrator-transport
```

```bash
uv run --env-file .env sound-receive
```

```bash
uv run --env-file .env mic-stream
```

先让 Sound 完成 sink 注册，再启动 Mic。Mic 收到 `mic.input.ready` 后才开始采集并发送 ASR/evidence control 事件；Mic 不发送 RTP。若进程在启动前失败，按下表检查配置：

正常的静音或无有效语音片段不会由 Mic 发送 `asr.final`，因此不会发起 Gate、LLM 或 TTS。

| 错误 | 原因与处理 |
| --- | --- |
| `config field is blank: ORCHESTRATOR_TRANSPORT_ADVERTISED_HOST` | `.env` 未加载，或缺少该变量；使用 `uv run --env-file .env orchestrator-transport` 并填写 loopback 地址。 |
| `MIC_ASR_ENDPOINT: response lacks text` | 确认 Mic ASR endpoint 返回带字符串 `text` 的 OpenAI-compatible transcription 响应。 |
| WebSocket `HTTP 401` | Mic 或 Sound 在回环模式仍发送 token；将其 `.env` 中的 `TRUSTED_LAN_TOKEN` 清空后重启客户端。 |
| `ORCHESTRATOR_WS_URL: must use WSS outside explicit loopback test mode` | Sound 缺少 `SOUND_ALLOW_LOOPBACK_WS=true`，或 URL 不是 `ws://localhost`、`ws://127.0.0.1` 或 `ws://[::1]`。 |

停止时按 Mic、Sound、Orchestrator 的反向顺序退出。不要把此配置复制到 systemd 或 LAN 主机。
