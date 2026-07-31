# Raspberry Girl 开发者文档

本文档描述整个 Raspberry Girl 工作区的开发视角。总体架构、协议和跨模块契约以 Orchestrator 为权威；模块内部行为在各模块仓库的开发者文档中维护。

## 项目概览

工作区包含五个独立仓库：`bitnp-raspberrygirl-vtuber-orchestrator`、`bitnp-raspberrygirl-vtuber-mic`、`bitnp-raspberrygirl-vtuber-sound`、`bitnp-raspberrygirl-vtuber-comments` 和 `bitnp-raspberrygirl-vtuber-frontend`。Orchestrator 是 hub，其他模块是 spoke。任何跨服务状态提交、provider 调用、命令派发和契约验证都必须经 Orchestrator。

## 技术栈

- Orchestrator、Mic、Sound、Comments：Python 3.12+、`uv`、`pytest`、`websockets`。Mic 和 Sound 使用 `sounddevice` 作为本地音频边界。
- Orchestrator 开发检查：`basedpyright`、`ruff`、JSON Schema fixture 验证、拓扑验证和 Frontend 契约验证。
- Frontend：Godot 4.6，主场景 `res://raspberry_girl.tscn`，配置项 `application/run/orchestrator_ws_url` 指向 Orchestrator。

## 项目架构

```text
Comments -- WSS audience.input --> Orchestrator <-- WSS control/result -- Frontend
Mic      -- WSS source control --> Orchestrator <-- WSS sink control ---- Sound
Mic      -- UDP L16 RTP -------> Orchestrator -- UDP generated L16 RTP -> Sound
```

Orchestrator 拥有 session state、revisioned event history、active turn、task registry、cancellation epoch、交互策略和 provider 边界。Mic、Comments、Sound 不感知业务策略，不持有跨服务状态。Frontend 只消费 Orchestrator 命令并返回受限结果。

## 数据流动关系

语音输入从 Mic 进入 Orchestrator 的 RTP ingress。现场语音交互链路启用时，Orchestrator 对 20 ms L16 RTP 帧做端点检测，约 600 ms 静音后提交并以 15 秒为单段上限，封装 16 kHz mono PCM WAV 给 ASR，再经 turn pipeline、LLM 和 TTS 生成新的 WAV，校验后重新 packetize 为 L16 RTP 发给 Sound。每个输出使用独立 packetizer 和生成 SSRC；新的有效 ASR final 会取消过期回答工作，已取消的 LLM/TTS 结果不得产生 RTP。原始 Mic RTP 不直接转发给 Sound。

评论输入由 Comments 以规范 envelope 提交为 `audience.input`。Frontend 只接收 Orchestrator 源的 caption、action、scene、presentation 等命令，演示命令完成后返回 `presentation.result`。所有迟到、超时、取消或被 supersede 的任务即使物理完成，也不能提交状态或产生副作用。

## 通信协议

规范协议只存放在 Orchestrator：

- `schemas/protocol/envelope.schema.json`
- `schemas/protocol/event-data.schema.json`
- `schemas/fixtures/valid/` 和 `schemas/fixtures/invalid/`

封闭 envelope 必须携带 schema version、event identity、source、time、`trace_id`、`session_id`、`seq` 和 typed `data`。RTP 媒体契约固定为 L16、16 kHz、mono、payload type 96、每帧 320 samples。客户端仓库只能引用 Orchestrator schema，不能复制 schema 或 fixture。

## 模块契约

- Orchestrator：唯一 session state writer；唯一协议权威；唯一 ASR/LLM/TTS provider 边界；唯一跨服务 reducer 和命令校验者。
- Mic：只向 Orchestrator 注册 RTP source，收到 matching `media.rtp.source.ready` 后才发送 UDP RTP。
- Sound：只向 Orchestrator 注册 RTP sink，只播放匹配 `media.stream.command` 的流，并报告 queued、playing、finished、cancelled、flush ack 等状态；只有精确关联的 `finished` 才能释放输出 lease。
- Comments：只向 Orchestrator 发送观众输入，不拥有平台生产接入的全功能边界。
- Frontend：只连接 Orchestrator，执行有限动作、表情、场景和演示控制映射。

## 模块行为

Orchestrator 的调度器把工作分为 reflex、interactive、deliberative 和 maintenance lane。反射类行为，如打断、TTS gate 和 RTP 输出 gate，不能等待 LLM、检索、MCP 或后台任务。

Mic 和 Sound 的媒体边界保持固定的 16 kHz mono PCM16/L16 RTP。Comments 保持回放和健康检查能力。Frontend 不参与 onsite audio loop；字幕 cue 在协议层准备就绪，但不要把同步字幕渲染描述为已完成能力。

## 关键技术细节

- 用户语音打断必须立即停止或 gate TTS 和生成 RTP，并取消被 supersede 的非必要任务。
- LLM 输出是不可信提案，动作、翻页、MCP 调用等必须转为 closed typed command 后再校验。
- mutable memory、immutable knowledge、session working memory 三者分离。
- 说话人 diarization 是 session-local 标签，不是身份；跨 session 说话人识别需要显式同意、模板保护和删除路径。
- MCP 调用必须 capability-scoped、deadline-bound、cancellable，并经 turn reducer 返回。

## 验证命令

从 Orchestrator 仓库运行：

```bash
uv run basedpyright
uv run ruff check src tests
python scripts/verify_protocol_schema.py
python scripts/verify_topology.py --sibling-root ..
python scripts/verify_vtuber_contract.py --frontend-path ../bitnp-raspberrygirl-vtuber-frontend
bash scripts/verify_workspace.sh --sibling-root ..
```

`verify_workspace.sh` 只组合 schema、topology 和 Frontend contract gate，不替代任何仓库的本地测试。各服务的本地命令由其用户文档维护。

## 本机回环联调

本机回环只用于同一台机器上的开发联调，和现场/LAN 部署是两套互斥配置。它使用未加密的 `ws://localhost`，因此所有监听与通告地址都必须是 loopback；不得把该配置暴露到局域网。完整变量、启动顺序和故障对照见[本机回环联调指南](local-loopback.zh-CN.md)。

服务进程不会自行读取 `.env`。开发时从每个服务仓库运行 `uv run --env-file .env <command>`；systemd 部署则通过 `EnvironmentFile` 注入变量。

## 部署

`orchestrator-transport` 是中心进程，监听认证 WSS 控制连接和 UDP RTP。生产环境使用 `/control` WSS endpoint、`TRUSTED_LAN_TOKEN`、仓库外预配的 TLS 证书、一个只读 PEM CA bundle 及私有 LAN 网络规则。Orchestrator、Mic、Sound 和 Comments 都设置 `ORCHESTRATOR_TLS_CA_PATH` 指向该 bundle；它可包含内部根证书和中间证书。Orchestrator 用它校验自托管 ASR、LLM、TTS 的 HTTPS endpoint，Mic、Sound、Comments 用它校验 Orchestrator WSS 证书。主机系统信任库可作为已安装相同 CA 时的替代，但不是部署契约。Mic 与 Sound 使用同一个 session ID 和 stream ID，且只连接 Orchestrator。具体挂载和环境文件见 [部署资产](../deploy/README.md)。

现场音频链路的启动顺序固定为：

```bash
uv run orchestrator-transport
uv run sound-receive
uv run mic-stream
```

该链路把 Mic L16 RTP 送入 Orchestrator，经过 ASR、LLM 和 TTS 后将生成的 L16 RTP 交给 Sound。它不会转发原始 Mic RTP，Frontend 不参与该音频部署。
