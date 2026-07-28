# 部署

无凭据开发请保留 `ORCHESTRATOR_LLM_PROVIDER=mock`。使用 OpenAI 兼容的 ASR、LLM 或 TTS 时，在部署环境中设置对应的 `ORCHESTRATOR_ASR_*`、`ORCHESTRATOR_LLM_*` 和 `ORCHESTRATOR_TTS_*` endpoint、model 与 API key 占位配置。vLLM Omni 声音克隆是 OpenAI 兼容的 TTS 扩展。不得提交真实密钥。

## 传输运行时

将 `orchestrator-transport` 作为 Orchestrator 所有的中心进程运行。它监听经过认证的 WSS 控制连接和 UDP RTP。Mic 与 Sound 只连接此中心。不要创建服务之间的 peer 连接，也不要把规范 schema 复制到此仓库之外。

部署环境必须设置 `.env.example` 中的全部传输变量：

- `ORCHESTRATOR_CONTROL_BIND_HOST` 和 `ORCHESTRATOR_CONTROL_BIND_PORT` 选择本地 WSS 监听地址。
- `ORCHESTRATOR_RTP_BIND_HOST` 和 `ORCHESTRATOR_RTP_BIND_PORT` 选择本地 UDP RTP 监听地址。
- `ORCHESTRATOR_TRANSPORT_ADVERTISED_HOST`、`ORCHESTRATOR_TRANSPORT_ADVERTISED_CONTROL_PORT` 和 `ORCHESTRATOR_TRANSPORT_ADVERTISED_RTP_PORT` 标识供 peer 配置使用的可达 LAN 端点。应设置为 Mic 和 Sound 能到达的地址和端口。只有存在对应转发规则时，它们才可以与 bind 设置不同。
- `TRUSTED_LAN_TOKEN` 是 WSS 握手共享的 Bearer token。真实值只能存放在部署密钥存储中，不得写入 `.env.example`、文档、源代码仓库或命令历史。
- `ORCHESTRATOR_CONTROL_TLS_CERT_PATH` 和 `ORCHESTRATOR_CONTROL_TLS_KEY_PATH` 是仓库外部预配的证书和私钥的只读路径。运行时自行加载该文件对。不要在此生成、嵌入或提交 TLS 材料。

生产环境始终使用 WSS。非 loopback 配置若缺少任一 TLS 路径或 `TRUSTED_LAN_TOKEN`，进程会拒绝启动。外部端口若转发到控制 bind 端口，必须保留 WSS TLS 连接，因为运行时拥有证书和私钥。控制和 RTP 监听器应限制在私有 LAN。只允许来自 Mic 和 Sound 网络的 TCP 流量访问可达控制端口，UDP 流量访问可达 RTP 端口。不要将任一监听器公开暴露，也不要把 Bearer token 当作防火墙策略的替代品。

`ORCHESTRATOR_TRANSPORT_ALLOW_LOOPBACK_WS=true` 是明确的测试例外。仅当控制 bind host、RTP bind host 和 advertised host 全部为 `127.0.0.1`、`::1` 或 `localhost` 时，它允许不使用 TLS 和 token 的明文 WS。它不是生产或 LAN 诊断模式。

## 路由与监听器就绪

仅当 UDP RTP 监听器和 WSS 控制监听器都已启动时，运行时才处于 listener ready 状态。它没有 stdin 或 stdout 控制协议，也没有 HTTP readiness 端点。应由服务管理器监管长期运行的进程，启动失败即表示未就绪。

一个 RTP 路由需要同一 session 和 stream 的已认证 source 注册与 sink 注册。第一个有效的 source RTP 包会固定 source UDP 端口。只有固定的 source 和 sink 路由同时存在时才会开始转发。规范的 source 与 sink ready 消息会被接受，但它们本身不能证明 RTP 已可流动。中心只转发 RTP V2、payload type 96、L16 的规范数据包，其他 UDP 输入会被丢弃。

## 启动与验证

部署系统提供环境变量并挂载外部预配的 TLS 文件后，使用以下命令启动进程：

```bash
uv run orchestrator-transport
```

部署前在 Orchestrator 仓库中运行以下检查：

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

loopback 集成检查只验证 WS 和 UDP 测试路径。它不能替代具有外部预配 TLS 文件和防火墙规则的私有 LAN WSS 部署。

## 现场讲解链路

只有真实现场链路才设置 `ORCHESTRATOR_MODE=onsite_explainer`。它要求 `ORCHESTRATOR_ASR_PROVIDER=openai_compatible`、`ORCHESTRATOR_LLM_PROVIDER=openai_compatible` 和 `ORCHESTRATOR_TTS_PROVIDER=vllm_omni`，以及 `.env.example` 中全部对应的 endpoint 和 model 变量。通过密钥注入提供各 provider 的 API key，其中 LLM API key 必填。还必须设置非空的 `ORCHESTRATOR_TTS_VOICE`、`ORCHESTRATOR_TTS_REF_AUDIO` 和 `ORCHESTRATOR_TTS_REF_TEXT`，声音参考 WAV 路径必须以只读方式挂载在源代码仓库之外。

每个已接受的 Mic 话语会先累积五十个 20 ms 规范 L16 帧，再将固定的 16 kHz 单声道 PCM WAV 发送给 ASR，取得 LLM 回答后发送给 vLLM-Omni TTS 扩展，验证其返回的固定 16 kHz 单声道 PCM WAV，最后向 Sound 发出生成的 L16 RTP。该模式不会转发原始 Mic RTP。

Mic 与 Sound 必须使用相同的 session ID 和 stream ID、已部署的 `wss://<host>/control` 端点及共享 trusted-LAN token。两者保持 mode-agnostic，且没有直接 peer 端点。Frontend 不属于此现场音频部署。有关密钥挂载、私有 LAN 防火墙规则、PortAudio 访问、启动与回滚顺序和实时生成 TTS 验收测试，请参阅 [systemd 部署包](../deploy/README.md)。
