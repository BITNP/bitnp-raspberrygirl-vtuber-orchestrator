# 树莓娘开发者文档

本文档描述整个树莓娘工作区的开发视角。总体架构、协议和跨模块契约以 Orchestrator 为权威；模块内部行为在各模块仓库的开发者文档中维护。

## 项目概览

工作区包含五个独立仓库：`bitnp-raspberrygirl-vtuber-orchestrator`、`bitnp-raspberrygirl-vtuber-mic`、`bitnp-raspberrygirl-vtuber-sound`、`bitnp-raspberrygirl-vtuber-comments` 和 `bitnp-raspberrygirl-vtuber-frontend`。Orchestrator 是 hub，其他模块是 spoke。任何跨服务状态提交、provider 调用、命令派发和契约验证都必须经 Orchestrator。

## 技术栈

- Orchestrator、Mic、Sound、Comments：Python 3.12+、`uv`、`pytest`、`websockets`。Mic 和 Sound 使用 `sounddevice` 作为本地音频边界。
- Orchestrator 开发检查：`basedpyright`、`ruff`、JSON Schema fixture 验证、拓扑验证和 Frontend 契约验证。
- Frontend：Godot 4.6，主场景 `res://main.tscn`；启动时从显式 JSON 配置文件读取连接参数，将校验后的值写入进程内 ProjectSettings 供控制客户端使用，不读取打包的部署凭据作为替代。

## 项目架构

```text
Comments -- WSS audience.input --> Orchestrator <-- WSS control/result -- Frontend
Mic      -- WSS input/ASR/evidence control --> Orchestrator <-- WSS sink control ---- Sound
Orchestrator -------------------------------------- UDP generated L16 RTP ---> Sound
```

Orchestrator 拥有 session state、revisioned event history、active turn、task registry、cancellation epoch、交互策略及 LLM/TTS provider 边界。Mic、Comments、Sound 不感知业务策略，不持有跨服务状态。Frontend 只消费 Orchestrator 命令并返回受限结果。

## 数据流动关系

### 行动与表达需求

语音（说话）本身是一种可选输出操作。树莓娘由 Brain 根据场景、上下文和当前能力自主选择语音、MCP、形象表情/动作、PPT 翻页或不操作；接受输入不意味着立即语音回复。允许静默执行 MCP、操作完成后再反馈、全程静默或接受信息而暂不表达。语音与评论入口必须一致执行提案的实际选择，轮次完成不应依赖 TTS，静默接纳也不应自动打断当前播放。共享需求及验收场景见[智能体行动与表达约定](../../.scratch/agent-response-choice/spec.md)。

知识库采用自动 RAG：Orchestrator 在首次 Brain 调用前，根据当前输入进行本地 BM25 检索，把相关摘录、来源和版本注入 `knowledge_references`。这属于上下文准备，不是 Brain 主动请求的操作，也不通过 MCP 调用；检索不会自动触发语音输出。

**实现状态：v2 已适配。** 版本化提案允许以空 speech 明确选择静默；初始与结果阶段统一执行。静默输入提交、操作执行、完成与维护独立于 TTS，独立形象动作经 `avatar.cue` 可信映射派发。下面的语音链路只用于实际请求 speech 的提案。

### 当前输入与媒体链路

Mic 在本地对 20 ms PCM16 帧进行 VAD、CAM++、端点检测，并将窗口提交给 OpenAI-compatible ASR；它在同一认证 control connection 发送 `asr.final`，`asr.partial` 仅用于诊断。Orchestrator 只接受已注册 stream、当前 session/epoch、未重放序列及合法 RTP 范围的 final，然后与评论共用单一 Brain 候选队列。Mic 没有 UDP RTP 输入路径。LLM/TTS 生成的音频经校验后 packetize 为 L16 RTP 发给 Sound。每个输出使用独立 packetizer 和生成 SSRC；新的已接受输入会取消过期回答工作，已取消的 LLM/TTS 结果不得产生 RTP。

评论输入由 Comments 以规范 envelope 提交为 `audience.input`。ASR final 和评论进入同一个每会话串行候选队列，容量为 16，语音优先；队列已满时，新语音可淘汰最旧的排队评论。Frontend 只接收 Orchestrator 源的 caption、action、scene、presentation 等命令，演示命令完成后返回 `presentation.result`。所有迟到、超时、取消或被替代的任务即使物理完成，也不能提交状态或产生副作用。

## 通信协议

规范协议只存放在 Orchestrator：

- `schemas/protocol/envelope.schema.json`
- `schemas/protocol/event-data.schema.json`
- `schemas/fixtures/valid/` 和 `schemas/fixtures/invalid/`

封闭 envelope 必须携带 schema version、event identity、source、time、`trace_id`、`session_id`、`seq` 和 typed `data`。RTP 媒体契约固定为 L16、16 kHz、mono、payload type 96、每帧 320 samples。客户端仓库只能引用 Orchestrator schema，不能复制 schema 或 fixture。

生产示例统一使用 `/control` 作为 WebSocket URL 路径；当前 listener 不按 URL path 分流，握手身份完全来自 bearer token 映射，消息权限再由已绑定角色和 session 所有权决定。Mic input、Sound sink 和 Frontend 注册可以创建 session；Comments 与 operator 只能使用已有 session。连接断开时移除该连接拥有的 route、ack 和 lease；无 owner、无活动工作且超过 TTL 的 session 在 sweep 时完整清理。

## 模块契约

- Orchestrator：唯一 session state writer、协议权威、Brain 与 LLM/TTS provider 边界，以及唯一跨服务 reducer 和命令校验者。
- Mic：唯一 VAD/endpoint/ASR provider 边界；只在认证 Orchestrator control connection 上注册 `mic.input.register`，提交 `asr.final` 与可选 `voice.evidence`，协议另支持诊断用 `asr.partial`。当前 endpoint-batched ASR 管线不产生 partial。Mic 不创建 RTP route，也不发送 UDP RTP。
- Sound：只向 Orchestrator 注册 RTP sink，只播放匹配 `media.stream.command` 的流，并报告 queued、playing、finished、cancelled、flush ack 等状态；只有精确关联的 `finished` 才能释放输出 lease。
- Comments：只向 Orchestrator 发送观众输入，不拥有平台生产接入的全功能边界。
- Frontend：只连接 Orchestrator，执行有限动作、表情、场景和演示控制映射。

## 模块行为

### 精简回复契约与异步任务

单一 Brain 同时完成输入取舍与回复，使用严格 `decision/speech/operation` 提案，不存在独立的 LLM Gate。v2 要求 `schema_version="2.0.0"`；`discard` 为空 speech 且无操作，`accept` 可以用空 speech 静默接纳，并可带至多一个具有独立 arguments 的非语音操作。无版本旧提案保持非空 speech 限制。speech 仅进入 TTS、context 和字幕，arguments 仅进入注册工具的 schema 校验与请求构造。畸形 JSON、未知 intent、非法参数或非法 cue 均无效果，也不进行文本回退或 JSON 修复。本地知识在首次 Brain 前完成有界检索；操作结果最多回填一次，最终 Brain 在 deliberative 分道返回无操作的可选 speech；工具超时也进入一次有界结果决策。

候选进入 Brain 前只执行确定性的低成本检查，例如单字符 ASR 噪声和最近回复回声；这些检查不生成内容，也不是另一个模型决策层。Brain 返回后，Orchestrator 再校验播放期间的打断语义、连接所有权、会话、重放状态、revision、操作和 cue。候选只有全部通过才原子创建正式 turn、推进取消代次并提交输入与 speech；被丢弃或校验失败的候选不会进入上下文。

ASR 候选进入 session admission queue 时，Orchestrator 用自己的 monotonic clock 冻结 `was_playing_1000ms_ago`，不使用 Mic 的进程时钟，也不在候选排到队首后重新计算。该值为 true 时，Brain 的 `accept` 仍是不可信提案；reducer 只允许包含明确停止、等待、纠正或切换话题措辞的 ASR 通过，其余统一以 `brain_playback_policy_violated` 丢弃。确定性回声检查只参考实际开始播放的文本及播放结束后 1 秒尾窗，使用精确文本片段或整体相似度至少 0.88 的近似复述匹配；单字符 ASR 噪声在 Brain 前丢弃，Brain 对不清晰 ASR 生成的复述或“请重复”回复也在正式 turn 前 fail-closed。comment 不受这些规则影响。Mic control 接收循环只负责协议校验和快速投递候选任务，不等待 Brain 完成，因此慢模型不会把后续 ASR 堵在 WebSocket 缓冲区外，也不会改变其入队时播放判定。

演示工具只在启动时配置非空 `ORCHESTRATOR_PPT_DECK_CATALOG` 后注册：`presentation.load` 只接受目录内 `deck_id`，`presentation.navigate` 只接受 1 到 10000 的整数 `page`，`presentation.play` 只接受空对象，且三者拒绝额外字段。Orchestrator 根据当前状态补入可信的 session、turn、command ID、deck version 和页码，模型参数不能覆盖这些字段。执行前再次验证实时 capability、revision、epoch 与当前 deck 前置条件；只有 session-owning Frontend 对精确 command ID 的一次回执可提交演示状态，错误 owner、重复或迟到回执均无效。Frontend 部署侧还需准备允许文稿的版本化页面目录；真实渲染失败不会提交演示状态。

回复可含 `<action name="..."/>` 和 `<expression name="..."/>`。动作 allowlist 为 `act_cute`、`emphasis`、`hello`；expression allowlist 为 `nod`（点头）、`shake_head`（摇头）、`wink`（单眼眨眼），播放旧前端录制的面捕参数序列。候选准入和最终回复阶段使用相同白名单；Orchestrator 拒绝含未知或非法控制标记的候选，TTS 接收去除合法 cue 后的文本。Frontend 使用 canonical `vtuber.caption.timeline.command` / `vtuber.caption.timeline.cancel` 事件按 `inline-cue/v1` 渲染字幕，通过 Live2D 驱动执行原生动作、录制面捕、眨眼和口型。录制面捕不会覆盖字幕驱动的嘴部开合，不开启实时摄像头。

LLM 使用 OpenAI-compatible Chat Completions。所有项目编写的系统、任务、记忆和压缩提示词使用中文，引用材料保留原文。生成参数支持全局值及 Brain、maintenance 两级覆盖，实际请求参数由 workload 与显式 provider 方言共同决定；完整变量、范围、继承顺序和 `omit` 语义见[用户文档](user.zh-CN.md#llm-生成参数)。Chat Template 与 reasoning parser 属于模型服务端职责，Orchestrator 发送 `system`/`user` messages，不在客户端重复套模板。`reasoning_content` 不作为回复正文；如果服务端把思考混入合法 JSON 的 `speech`，当前结构校验不能可靠识别无标记的思考文本。

正式 turn 内的 Brain、MCP、TTS、flush、字幕投递、记忆提取和上下文压缩由 `TaskRegistry` 生命周期管理；尚未形成 turn 的首次 Brain 候选由每会话候选队列及独立 cancellable provider task 跟踪。正式任务先经 `ADMITTED → QUEUED → RUNNING → SUCCEEDED`，队列反压时仍停在 admission 边界并撤回；取消会先短暂进入 `CANCELLING` 关闭结果栅栏，再成为不可复用的 `CANCELLED` tombstone。字幕 timeline 在首个 RTP 帧获准后登记为短生命周期 interactive 任务；投递前重新核验 session、turn、revision、数据快照、epoch、deadline 与能力，投递成功也须经 reducer 提交。Sound replacement 在发出 flush 前也登记 interactive 任务，ACK 只会暂存新 lease；仅当该任务仍当前且 reducer 接受切换结果时才提交新 lease。ACK 后取消、过期或结果拒绝会回滚到旧 lease，使旧音频继续播放。前端不可用只使该字幕任务失败，绝不回滚音频。任务结果提交前需校验 session、turn、revision、epoch 和 deadline；取消先关闭结果栅栏，再取消 provider，因此迟到结果不得产生媒体、上下文、记忆或前端效果。replacement TTS 必须持有首个有效 RTP 帧并等待 Sound flush ACK，失败时原播放保持不变。

Orchestrator 的调度器把工作分为 reflex、interactive、deliberative 和 maintenance lane。反射类行为，如打断、TTS gate 和 RTP 输出 gate，不能等待 LLM、检索、MCP 或后台任务。

已验证 speech 的 TTS、Sound flush 和字幕投递使用 `validated_speech` 数据依赖：
记忆修订和上下文压缩可以独立完成，不会使这些已确定的媒体内容失效。
会话 revision、turn、epoch、deadline、能力，以及身份、同意和知识版本仍须通过校验。
Brain、工具和维护任务默认保留完整数据快照检查。

ASR 回声过滤只参考通过首帧准入的实际播放文本，播放结束后保留 1 秒尾窗；
未播放的字幕及历史对话不作为回声证据。精确文本片段或整体相似度至少 0.88
的近似复述才可能被过滤，仅有相同主题不能判定回声。澄清防护只匹配直接向当前
用户求重复的句首表达，跳过成对引号内的例句；模糊语义仍交给 Brain 判定。
英文打断匹配保留单词边界，因此 `hold on` 和 `please stop now` 均可识别。

Brain 和维护的 JSON HTTP 响应分块读取，解码后的完整响应体上限为 1 MiB，
包含 `reasoning_content` 和其他元数据。超限立即关闭响应流并拒绝结果，
不记录超限正文；完整且未超限的响应仍按 DEBUG 日志规则记录。

### TurnCoordinator 状态与执行信封

每个接受的输入由协调器从不可变快照生成内部 `ExecutionEnvelope`。它固定
`session_id`、`turn_id`、`segment_id`、revision、cancellation epoch、deadline、媒体
替换策略以及动作/表情 allowlist；这些字段绝不来自模型。`SessionRuntime` 只能将
已经被 `SessionScheduler` 接纳的 `turn_id` 与 epoch 交给 `TurnCoordinator`，不能由
provider 回调自行推进状态。正常状态为
`QUEUED → REASONING → WAITING_TOOL（可选）→ SYNTHESIZING → CUTOVER_PENDING（仅替换）→ PLAYING → COMPLETED`，
任一未完成状态都可因新输入、deadline、能力撤销或会话结束进入 `CANCELLED`。旧 epoch、
旧 turn 或错误 phase 的回调只记录诊断，绝不推进状态或产生效果。替换期间当前逻辑 turn
可以处在准备状态，但旧物理 playback lease 仍由 `SchedulerOutputFence` 保留，直到 flush
task 的结果栅栏允许切换。

首次 Brain 候选不创建正式 turn、不推进取消代次，也不停止当前播放；接受后才原子创建 turn 并立即提交输入与首次 speech。首次 Brain 的 provider coroutine 由候选队列跟踪；正式 turn 之后的 Brain、受控工具、TTS、记忆提取和上下文压缩登记为 task。首次与
最终 Brain 最多各一次；最终调用禁止 operation。首次 speech 合成与唯一工具并行，工具、LLM 或维护 provider 的返回先
经过 task/revision/data-snapshot/epoch/deadline 栅栏，再允许创建下一任务或提交结果。
经过校验的 speech 在 TTS 前写入 transient context；终态 speech 后才会安排 memory/compaction maintenance
任务。每段替换播放必须先获首帧和匹配的 Sound flush ACK；成功时先取消旧字幕 timeline，
失败则旧音频与旧 timeline 都保持。`PLAYING → COMPLETED` 只能由 Sound 已通过输出
lease 校验的 `finished` 事件触发；TTS provider 完成或重复/过期 finished 都不能结束逻辑 turn。
如果 replacement flush 被拒绝、超时或失效，`TurnCoordinator` 会恢复已保留旧 lease 的
`PLAYING` 状态；新 turn 不得写入 context、memory 或 timeline。

独立 Gate、shadow/execute 模式和现场回退均已删除。缺少 Brain coordinator 时输入 fail-closed；现场 callback 对已丢弃和已接受输入都返回 handled，不能回落到旧 ASR/LLM 路径。

Mic 和 Sound 的媒体边界保持固定的 16 kHz mono PCM16/L16 RTP。Comments 当前只提供 JSONL 回放和配置健康检查，不是直播平台生产接入器。Frontend 不参与 onsite audio loop，但已实现音频获准后的逐字字幕 timeline、口型和 cue 动作；它通过受控本地页面目录实现真实 deck 渲染。

## 关键技术细节

### 上下文与记忆边界

上下文的 512-token 预算采用 UTF-8 字节数作为保守上界，适用于中文、混合语言、
标点及无空格长文本；这不是模型 tokenizer 的精确计数。已有摘要与近期正文共享预算。
超预算时，提示词最多为摘要分配一半预算，保留可容纳的近期原文；完整源快照仍交给独立
压缩任务，只有源快照保持不变时才能原子替换摘要。压缩失败不会将超预算正文注入 Brain。

记忆提取器的类别声明不能授权持久化。可信代码检查 snake_case key、单行正文、
长度、置信度及中英文敏感类别和常见联系方式、凭据格式，并对 Unicode 做规范化。
规则会保守拒绝涉及敏感类别的材料；它不是通用语义隐私分类器，仍可能误拒或漏检隐含表达。
受保护候选不记录 key/value 正文。每会话最多 32 条记忆，key/value 合计不超过
16 KiB UTF-8；单条 key 最多 128 字符、value 最多 512 字符。容量满时拒绝新增候选，
不自动淘汰用户记忆；删除后可继续写入。

`memory.md` 和兼容 JSON 存储使用版本 2 格式。新增、替换、删除的最近 64 条审计记录
与记忆在同一文件原子替换，包含版本、时间、来源及前后内容 SHA-256；最近 16 条冲突
记录另保留已校验正文。删除 key 同时删除其冲突正文，审计只留下摘要。审计位于机器状态块，
不会注入 Brain。存储读写上限为 256 KiB，读取旧文件也会重新验证条目；不安全或超限文件
会被拒绝加载并保留原文件，不会自动迁移或删除。写盘失败不推进内存状态。会话结束或 TTL
回收会删除记忆及审计。

- 播放期间的语音候选只有明确表达停止、纠正或切换话题并通过 Brain 与 reducer 校验后才成为替换轮次；候选评估期间保留当前播放，成功切换后取消被替代任务。
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

## 受信任局域网明文联调

显式设置 `ORCHESTRATOR_TRANSPORT_ALLOW_LOOPBACK_WS=true` 时，control listener 可在 loopback 或受信任局域网地址上使用 `ws://`，且不加载 control TLS certificate/key。该模式仍强制 Mic、Sound、Comments、Frontend 和 operator 使用彼此不同的角色 token；各客户端还必须显式开启自身的 `*_ALLOW_LOOPBACK_WS` 开关。完整变量、启动顺序和风险说明见[受信任局域网明文联调指南](local-loopback.zh-CN.md)。

服务进程不会自行读取 `.env`。开发时从每个服务仓库运行 `uv run --env-file .env <command>`；systemd 部署则通过 `EnvironmentFile` 注入变量。

## 部署

`orchestrator-transport` 是中心进程，监听认证 WSS 控制连接和 UDP RTP。生产环境使用 `/control` WSS endpoint，并分别配置 `ORCHESTRATOR_MIC_CONTROL_TOKEN`、`ORCHESTRATOR_SOUND_CONTROL_TOKEN`、`ORCHESTRATOR_COMMENTS_CONTROL_TOKEN`、`ORCHESTRATOR_FRONTEND_CONTROL_TOKEN` 和 `ORCHESTRATOR_OPERATOR_CONTROL_TOKEN`；各 peer 仍从自己的 `TRUSTED_LAN_TOKEN` 发送对应角色的值。五个值必须互不相同。部署还需在仓库外预配 TLS 证书、只读 PEM CA bundle、32 字节 AES voice-template key 及私有 LAN 网络规则。Orchestrator、Mic、Sound 和 Comments 都设置 `ORCHESTRATOR_TLS_CA_PATH` 指向该 bundle；它可包含内部根证书和中间证书。Orchestrator 用它校验自托管 LLM、TTS 的 HTTPS endpoint，Mic、Sound、Comments 用它校验 Orchestrator WSS 证书。Mic 的 ASR endpoint/model/credentials 只在 Mic 的环境中配置。Mic 与 Sound 使用同一个 session ID 和 stream ID，且只连接 Orchestrator。具体挂载和环境文件见 [部署资产](../deploy/README.md)。

现场音频链路的启动顺序固定为：

```bash
uv run orchestrator-transport
uv run sound-receive
uv run mic-stream
```

该链路由 Mic 产生 ASR final，Orchestrator 经单一 Brain 和 TTS 后将生成的 L16 RTP 交给 Sound。它不会转发原始 Mic RTP，Frontend 不参与该音频部署。

受众队列中的输入在出队并构建 Brain 快照时确定 revision，前序输入正常提交不会使排队输入失效。Brain 调用期间若 revision 改变，或连接所有权、会话、重放校验失效，仍拒绝该结果。

## 知识库和外部 MCP

生产装配在 transport_app 中一次构建只读中文 BM25 知识快照并注入所有会话。外部 MCP 使用原生异步 Streamable HTTP 请求，白名单与可信意图由本地配置登记。实现、限额和使用方式见 [知识库与 MCP](knowledge-mcp.zh-CN.md)。

### 可选表达契约与回执

Brain 新提案显式使用 `schema_version: "2.0.0"`，规范 schema 位于 `schemas/brain/response-proposal-v2.schema.json`。例如 `{"schema_version":"2.0.0","decision":"accept","speech":"","operation":null}` 表示静默接纳；operation 可请求允许的单个操作，操作前后 speech 各自独立决定。旧版无版本提案仍要求 accept 带有效语音，未知版本拒绝。所有入口均调度已请求语音，静默不创建 TTS、音频或字幕，不打断既有播放。最终 Brain 使用 deliberative lane，工具失败或超时也提供一次真实的最终决策机会。

独立 `avatar.cue` 经协议 1.2.0 命令和所属 Frontend 的匹配结果确认；成功代表动作已被驱动接纳，拒绝/超时不冒充成功。Sound 断线清理输出租约、切换状态和语音任务，保留输出 epoch 高水位及独立 Mic input_epoch。PPT 的部署准备、静态页面限制见 Frontend 用户文档。

回复提案的完整编排仅由 `SessionRuntime` 持有；`AsyncResponseCoordinator` 提供分步 provider 访问和可信映射，不提供另一套完整流程。行为测试通过受众输入入口验证 Brain 次数、操作结果和过期拒绝。
