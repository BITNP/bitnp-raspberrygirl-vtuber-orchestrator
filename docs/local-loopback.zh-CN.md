# 受信任局域网明文联调指南

本指南用于受控局域网或单机上的 Mic、Orchestrator、Sound、Comments 和 Frontend 联调。显式开启后，control plane 使用未加密的 `ws://`，但仍强制使用各角色独立的 bearer token。该模式没有机密性，任何能监听局域网流量的设备都可能看到 token 和完整控制载荷；不得用于访客网络、共享网络或公网。

文件名保留 `local-loopback.zh-CN.md` 以兼容已有链接，但开关现在允许非 loopback 的受信任局域网地址。

## 前提

- 每个 Python 服务仓库分别完成 `uv sync --locked`。
- 每个服务都有自己的未提交 `.env`，并用 `uv run --env-file .env <command>` 显式加载。
- 为 Mic、Sound、Comments、Frontend 和 operator 生成五个不同的高熵 token；客户端只持有自身角色的 token。
- Mic 与 Sound 使用相同的 session ID 和 stream ID。
- 防火墙只允许受信任主机访问 Orchestrator TCP control 端口；只有远程 Sound 需要 RTP UDP 路由。

## Orchestrator

```dotenv
ORCHESTRATOR_CONTROL_BIND_HOST=0.0.0.0
ORCHESTRATOR_CONTROL_BIND_PORT=8443
ORCHESTRATOR_RTP_BIND_HOST=0.0.0.0
ORCHESTRATOR_RTP_BIND_PORT=5004
ORCHESTRATOR_TRANSPORT_ADVERTISED_HOST=orchestrator.lan
ORCHESTRATOR_TRANSPORT_ADVERTISED_CONTROL_PORT=8443
ORCHESTRATOR_TRANSPORT_ADVERTISED_RTP_PORT=5004
ORCHESTRATOR_TRANSPORT_ALLOW_LOOPBACK_WS=true
ORCHESTRATOR_CONTROL_TLS_CERT_PATH=
ORCHESTRATOR_CONTROL_TLS_KEY_PATH=
ORCHESTRATOR_MIC_CONTROL_TOKEN=<mic-role-token>
ORCHESTRATOR_SOUND_CONTROL_TOKEN=<sound-role-token>
ORCHESTRATOR_COMMENTS_CONTROL_TOKEN=<comments-role-token>
ORCHESTRATOR_FRONTEND_CONTROL_TOKEN=<frontend-role-token>
ORCHESTRATOR_OPERATOR_CONTROL_TOKEN=<operator-role-token>
```

`0.0.0.0` 只用于 bind。`ORCHESTRATOR_TRANSPORT_ADVERTISED_HOST` 必须是 Sound 可达的实际 IP 或 DNS 名，不能是 `0.0.0.0`。

## Mic

```dotenv
ORCHESTRATOR_WS_URL=ws://orchestrator.lan:8443/control
MIC_ALLOW_LOOPBACK_WS=true
TRUSTED_LAN_TOKEN=<mic-role-token>
BITNP_SESSION_ID=session-onsite-001
BITNP_MIC_STREAM_ID=onsite-primary
MIC_ASR_ENDPOINT=http://127.0.0.1:8090/v1
MIC_ASR_MODEL=<openai-compatible-asr-model>
```

ASR 由 Mic 调用，因此 `127.0.0.1` 表示 Mic 所在主机。远程 ASR 应通过受控 LAN、VPN 或 HTTPS 暴露。

## Sound

```dotenv
ORCHESTRATOR_WS_URL=ws://orchestrator.lan:8443/control
SOUND_ALLOW_LOOPBACK_WS=true
TRUSTED_LAN_TOKEN=<sound-role-token>
SOUND_RTP_BIND_HOST=0.0.0.0
SOUND_RTP_BIND_PORT=5006
SOUND_RTP_ADVERTISED_HOST=sound.lan
SOUND_SESSION_ID=session-onsite-001
SOUND_RTP_STREAM_ID=onsite-primary
```

## Comments

```dotenv
ORCHESTRATOR_WS_URL=ws://orchestrator.lan:8443/control
COMMENTS_ALLOW_LOOPBACK_WS=true
TRUSTED_LAN_TOKEN=<comments-role-token>
```

## Frontend

`frontend-config.json`：

```json
{
  "orchestrator_ws_url": "ws://orchestrator.lan:8443/control",
  "orchestrator_tls_ca_path": "",
  "orchestrator_session_id": "session-onsite-001"
}
```

进程环境：

```dotenv
FRONTEND_ALLOW_LOOPBACK_WS=true
ORCHESTRATOR_FRONTEND_CONTROL_TOKEN=<frontend-role-token>
```

## 启动与验证

依次启动 Orchestrator、Sound、Frontend/Comments，最后启动 Mic：

```bash
uv run --env-file .env orchestrator-transport
uv run --env-file .env sound-receive
uv run --env-file .env mic-stream
```

Mic 收到 `mic.input.ready` 后才开始采集。Mic 不发送 RTP；Orchestrator 只向注册的 Sound sink 发送 RTP。

| 错误 | 原因与处理 |
| --- | --- |
| `config field is blank: ORCHESTRATOR_MIC_CONTROL_TOKEN` | 明文模式仍要求五个独立角色 token；补齐 Orchestrator token。 |
| `ORCHESTRATOR_WS_URL: must use WSS unless trusted-LAN insecure WS is explicitly enabled` | 在对应客户端设置 `MIC_`、`SOUND_` 或 `COMMENTS_ALLOW_LOOPBACK_WS=true`。 |
| WebSocket `HTTP 401` | 客户端 `TRUSTED_LAN_TOKEN` 与 Orchestrator 对应角色 token 不匹配。 |
| Frontend `unsafe_credential_transport` | 设置 `FRONTEND_ALLOW_LOOPBACK_WS=true`，并提供 Frontend 专属 token。 |

切回安全部署时，把所有 `*_ALLOW_LOOPBACK_WS` 设为 `false`，使用 `wss://`、TLS certificate/key 和 CA bundle。角色 token 在两种模式下都不能省略或复用。
