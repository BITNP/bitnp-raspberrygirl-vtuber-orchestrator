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
Mic      -- WSS input/ASR/evidence control --> Orchestrator <-- WSS sink control ---- Sound
Orchestrator -------------------------------------- UDP generated L16 RTP ---> Sound
```

Orchestrator 拥有 session state、revisioned event history、active turn、task registry、cancellation epoch、交互策略及 LLM/TTS provider 边界。Mic、Comments、Sound 不感知业务策略，不持有跨服务状态。Frontend 只消费 Orchestrator 命令并返回受限结果。

## 数据流动关系

Mic 在本地对 20 ms PCM16 帧进行 VAD、CAM++、端点检测，并将窗口提交给 OpenAI-compatible ASR；它在同一认证 control connection 发送 `asr.final`，`asr.partial` 仅用于诊断。Orchestrator 只接受已注册 stream、当前 session/epoch、未重放序列及合法 RTP 范围的 final，然后与评论共用 Gate、队列和 Brain。Mic 没有 UDP RTP 输入路径。LLM/TTS 生成的音频经校验后 packetize 为 L16 RTP 发给 Sound。每个输出使用独立 packetizer 和生成 SSRC；新的有效 ASR final 会取消过期回答工作，已取消的 LLM/TTS 结果不得产生 RTP。

评论输入由 Comments 以规范 envelope 提交为 `audience.input`。Frontend 只接收 Orchestrator 源的 caption、action、scene、presentation 等命令，演示命令完成后返回 `presentation.result`。所有迟到、超时、取消或被 supersede 的任务即使物理完成，也不能提交状态或产生副作用。

## 通信协议

规范协议只存放在 Orchestrator：

- `schemas/protocol/envelope.schema.json`
- `schemas/protocol/event-data.schema.json`
- `schemas/fixtures/valid/` 和 `schemas/fixtures/invalid/`

封闭 envelope 必须携带 schema version、event identity、source、time、`trace_id`、`session_id`、`seq` 和 typed `data`。RTP 媒体契约固定为 L16、16 kHz、mono、payload type 96、每帧 320 samples。客户端仓库只能引用 Orchestrator schema，不能复制 schema 或 fixture。

## 模块契约

- Orchestrator：唯一 session state writer、协议权威、Gate/Brain 与 LLM/TTS provider 边界，以及唯一跨服务 reducer 和命令校验者。
- Mic：唯一 VAD/endpoint/ASR provider 边界；只在认证 Orchestrator control connection 上注册 `mic.input.register`，并提交 `asr.partial`、`asr.final` 与可选 `voice.evidence`。Mic 不创建 RTP route，也不发送 UDP RTP。
- Sound：只向 Orchestrator 注册 RTP sink，只播放匹配 `media.stream.command` 的流，并报告 queued、playing、finished、cancelled、flush ack 等状态；只有精确关联的 `finished` 才能释放输出 lease。
- Comments：只向 Orchestrator 发送观众输入，不拥有平台生产接入的全功能边界。
- Frontend：只连接 Orchestrator，执行有限动作、表情、场景和演示控制映射。

## 模块行为

### 精简回复契约与异步任务

新回复路径使用 `{"reply":"...","intent":"answer"}`：模型不得生成状态操作、媒体命令、MCP 参数或记忆 patch。无法解析的输出按普通文本回复处理，不触发 JSON 修复请求。工具 intent 必须由 Orchestrator 的可信 `IntentSpec` 映射为参数，并且工具观察返回后只允许一次最终 `answer` 回复。

回复可含 `<action name="..."/>` 和 `<expression name="..."/>`。Orchestrator 只保留 allowlist 内的标记；TTS 接收去标记文本。未来 Frontend 使用 canonical `vtuber.caption.timeline.command` / `vtuber.caption.timeline.cancel` 事件按 `inline-cue/v1` 渲染字幕与 cue。

LLM、MCP、TTS、flush、字幕投递、记忆提取和上下文压缩必须由 `TaskRegistry` 生命周期管理。任务先经 `ADMITTED → QUEUED → RUNNING → SUCCEEDED`，队列反压时仍停在 admission 边界并撤回；取消会先短暂进入 `CANCELLING` 关闭结果栅栏，再成为不可复用的 `CANCELLED` tombstone。字幕 timeline 在首个 RTP 帧获准后登记为短生命周期 interactive 任务；投递前重新核验 session、turn、revision、数据快照、epoch、deadline 与能力，投递成功也须经 reducer 提交。Sound replacement 在发出 flush 前也登记 interactive 任务，ACK 只会暂存新 lease；仅当该任务仍当前且 reducer 接受切换结果时才提交新 lease。ACK 后取消、过期或结果拒绝会回滚到旧 lease，使旧音频继续播放。前端不可用只使该字幕任务失败，绝不回滚音频。任务结果提交前需校验 session、turn、revision、epoch 和 deadline；取消先关闭结果栅栏，再取消 provider，因此迟到结果不得产生媒体、上下文、记忆或前端效果。replacement TTS 必须持有首个有效 RTP 帧并等待 Sound flush ACK，失败时原播放保持不变。

Orchestrator 的调度器把工作分为 reflex、interactive、deliberative 和 maintenance lane。反射类行为，如打断、TTS gate 和 RTP 输出 gate，不能等待 LLM、检索、MCP 或后台任务。

### TurnCoordinator 状态与执行信封

每个接受的输入由协调器从不可变快照生成内部 `ExecutionEnvelope`。它固定
`session_id`、`turn_id`、`segment_id`、revision、cancellation epoch、deadline、媒体
替换策略以及动作/表情 allowlist；这些字段绝不来自模型。正常状态为
`QUEUED → REASONING → WAITING_TOOL（可选）→ SYNTHESIZING → PLAYING → COMPLETED`，
任一未完成状态都可因新输入、deadline、能力撤销或会话结束进入 `CANCELLED`。

初始 LLM、受控工具、最终 LLM、TTS、记忆提取和上下文压缩分别登记为 task。初始与
最终 LLM 最多各一次；最终调用仅允许 `answer`。工具、LLM 或维护 provider 的返回先
经过 task/revision/data-snapshot/epoch/deadline 栅栏，再允许创建下一任务或提交结果。
音频首帧被接受后才写入 transient context，并才会安排 memory/compaction maintenance
任务。替换播放必须先获首帧和匹配的 Sound flush ACK；成功时先取消旧字幕 timeline，
失败则旧音频与旧 timeline 都保持。

迁移期间通过 `ORCHESTRATOR_RESPONSE_EXECUTION_MODE` 按 session 选择
`legacy_execute`、`new_shadow` 或 `new_execute`（默认）。`new_shadow` 运行新模型、
最小 proposal 解析和 cue 编译，并记录脱敏诊断；它绝不执行 MCP、TTS、Frontend、
context 或 memory 写入。若影子模式缺少新协调器，运行时 fail-closed 拒绝输入，不能回落
到 legacy 副作用路径。
`new_execute` 若缺少 Gate 或 ResponseCoordinator 也同样 fail-closed；不会隐式构造
mock AgentPlan pipeline 或回退到 legacy 执行。`legacy_execute` 仅允许显式迁移/回滚配置。

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

`orchestrator-transport` 是中心进程，监听认证 WSS 控制连接和 UDP RTP。生产环境使用 `/control` WSS endpoint、`TRUSTED_LAN_TOKEN`、仓库外预配的 TLS 证书、一个只读 PEM CA bundle 及私有 LAN 网络规则。Orchestrator、Mic、Sound 和 Comments 都设置 `ORCHESTRATOR_TLS_CA_PATH` 指向该 bundle；它可包含内部根证书和中间证书。Orchestrator 用它校验自托管 LLM、TTS 的 HTTPS endpoint，Mic、Sound、Comments 用它校验 Orchestrator WSS 证书。Mic 的 ASR endpoint/model/credentials 只在 Mic 的环境中配置。Mic 与 Sound 使用同一个 session ID 和 stream ID，且只连接 Orchestrator。具体挂载和环境文件见 [部署资产](../deploy/README.md)。

现场音频链路的启动顺序固定为：

```bash
uv run orchestrator-transport
uv run sound-receive
uv run mic-stream
```

该链路由 Mic 产生 ASR final，Orchestrator 经 Gate、LLM 和 TTS 后将生成的 L16 RTP 交给 Sound。它不会转发原始 Mic RTP，Frontend 不参与该音频部署。
