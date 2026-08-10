# Raspberry Girl VTuber Orchestrator

Orchestrator 是 Raspberry Girl 的中心仓库，拥有规范协议、会话状态、调度、LLM/TTS 边界和跨模块契约验证。Mic 负责 VAD、CAM++、端点检测和 ASR，并通过认证 control connection 提交结构化结果；Mic 不向 Orchestrator 发送 RTP。Orchestrator 的单一 Brain Pipeline 是唯一业务决策入口，Orchestrator 也是唯一状态写者与 effect 派发者，并只向 Sound 输出 TTS RTP。

现场音频由 Orchestrator 的输出 lease 裁决：每个完成的 TTS 响应使用独立 RTP packetizer 和 epoch/SSRC；只有携带精确 turn、segment 与 epoch 的 Sound `finished` 回报才能释放 lease。完成播放不会注销 Mic/Sound 路由，因此下一轮仍需经新的调度授权，旧 RTP 不会恢复。

生产演示操作只在配置非空 `ORCHESTRATOR_PPT_DECK_CATALOG` 时向 Brain 暴露。目录是最多 32 个受控 deck ID 的逗号分隔列表，不接受路径；`load`、`navigate`、`play` 分别使用固定的 `deck_id`、有界 `page` 和空参数 schema，并且必须由当前 session 的 Frontend 精确回执后才提交状态。

## 核心技术与项目优势

Raspberry Girl 不是把 ASR、LLM、TTS 和虚拟形象简单串联起来的 Demo，而是一套面向真实现场的事件驱动智能体中枢。它用同一个角色、同一份会话状态和同一条决策链路承接会议演示、展台讲解、虚拟直播与评论互动，让“听见、理解、行动、表达”形成可追踪、可取消、可验证的闭环。

```text
Mic 语音 ─┐                         ┌─ LLM / 本地知识 / MCP
          ├─ 统一输入 → Gate → Brain ┼─ TTS → L16 RTP → Sound
Comments ─┘             │           └─ 字幕 / 动作 / 演示 → Frontend
                        └─ Session Scheduler / Reducer / Memory
```

### 一个 Brain，统一理解所有现场输入

语音 ASR final 与观众评论会先归一化为统一的 audience input，再经过轻量相关性 Gate 和 session-local admission queue 进入同一个 Brain Pipeline。语音拥有更高的交互优先级，队列容量和请求大小均有明确上限；慢模型不会反向堵塞控制连接，无关评论也不会消耗完整推理链路。

这意味着角色不会因为输入来自麦克风、弹幕或演示现场就表现出割裂的“多个大脑”。所有渠道共享当前任务、对话上下文、记忆、知识引用、媒体状态和演示状态，回答更连贯，角色感也更稳定。

### 为实时交互设计的四级调度

Orchestrator 将任务分为 `reflex`、`interactive`、`deliberative` 和 `maintenance` 四条有界 lane：打断与媒体 gate 走毫秒敏感的反射路径；短回复和播放控制走交互路径；深度回答、检索与 MCP 进入审慎路径；记忆提取和上下文压缩在维护路径独立执行。所有慢任务都绑定 session、turn、revision、cancellation epoch 与 deadline。

用户再次开口时，系统可以立即 gate 过期输出并取消旧任务；即使 provider 稍后返回结果，reducer 也会因为版本或 epoch 已过期而拒绝副作用。相比“请求发出就只能等完”的普通流水线，这套机制能在多人、多输入和网络抖动下保持响应性，并避免旧回答突然插播。

### 不制造尴尬静音的流式语音切换

TTS 音频以 16 kHz、单声道 L16 RTP 输出，每段响应拥有独立的 packetizer、SSRC、command ID 和 cancellation epoch。替换正在播放的回答时，旧音频会继续播放，直到新 TTS 已产生首个有效 20 ms 音频帧，且 Sound 接受精确匹配的 flush；如果合成失败、任务过期或 flush 被拒绝，原播放保持不变。

这种“先准备、后切换”的双重栅栏避免了常见的抢断空窗，同时又能阻止重复包、迟到回执或旧流恢复播放。对听众而言，交互更接近真人自然接话，而不是频繁停顿的语音机器人。

### LLM 提建议，可信系统做决定

Brain 输出始终被视为不可信的 typed proposal。动作、表情、PPT 和 MCP 请求必须经过意图映射、严格参数 schema、capability allowlist、实时 revision/epoch、deadline 与命令前置条件校验，才能由 Orchestrator 派发。演示状态只在 session-owning Frontend 返回精确 command ID 的成功结果后提交。

MCP 同样使用静态的 server/tool/capability 白名单，并限制超时、请求和响应大小。外部结果不会直接触发效果，而是被压缩为带来源、状态和摘要指纹的有界 observation，再交回 Brain 生成最终回复。由此既保留智能体调用工具和控制舞台的能力，也把提示注入、幻觉指令和越权操作挡在可信边界之外。

### 三层上下文，让角色既连贯又可控

- **Transient Context** 保存 reducer 已接受的输入、最终回复和成功工具摘要；partial、discard、cancelled 或 stale 内容不会污染对话。超出预算时通过带源 revision/hash 的独立任务原子压缩。
- **Mutable Memory** 以 session 隔离、版本化的人类可读文档保存经过验证的长期信息，写入需要通过来源、置信度、敏感类别、冲突和 base revision 检查。
- **Immutable Knowledge** 使用 LlamaIndex 在启动时加载受控的本地 `.md`、`.txt` 和 `.json` 语料，携带 corpus/index revision，只检索、不被网络内容或运行时对话改写。

三者分离避免了“聊天记录、用户记忆和权威知识混成一团”：角色能记住必要信息、延续当前话题，也能保持资料来源稳定，并在 session 结束或过期时彻底清理会话数据。

### 从连接身份到每次副作用的纵深防护

Mic、Sound、Comments、Frontend 和 operator 使用不同的角色凭据连接 TLS WebSocket；权限来自已认证连接，而不是消息里自称的 `source`。Orchestrator 还会验证 session 所有权、注册关系、序列与重放状态，并将控制帧和 session identifier 限制在规范边界内。

每次状态写入和外部效果都保留 trace、session、sequence、turn 与 segment 关联。结合规范 JSON Schema、跨仓库拓扑验证、Frontend 契约验证和严格静态检查，系统既便于定位现场问题，也能持续防止模块绕过中心中枢形成隐蔽的数据通路。

### 可组合，而不是堆砌“产品模式”

同一套能力可以按部署环境组合：现场讲解可启用语音、知识与 PPT；虚拟直播可接入评论、动作和字幕；会议助手可侧重语音、阅读与记忆。能力通过显式配置和 allowlist 开放，不需要复制三套互不兼容的业务流程。

最终带来的核心价值是：

- **更像真人**：能被打断、能自然接续、输入渠道之间保持一致人格和上下文。
- **更适合现场**：有界队列、deadline、取消栅栏和无静音切换共同应对突发输入与服务抖动。
- **更安全可控**：模型不能直接写状态或执行效果，工具、演示和媒体操作都有可信校验边界。
- **更容易扩展**：ASR、LLM、TTS、知识库、MCP 和前端能力均通过清晰契约接入，核心状态机不随 provider 更换而重写。
- **更容易运维**：统一协议、完整关联日志、版本化状态和跨仓库验证让问题可复现、变更可审计。

## 进一步阅读

- [用户文档](docs/user.zh-CN.md)
- [开发者文档](docs/developer.zh-CN.md)

## 模块文档

- [Mic](../bitnp-raspberrygirl-vtuber-mic/README.md)
- [Sound](../bitnp-raspberrygirl-vtuber-sound/README.md)
- [Comments](../bitnp-raspberrygirl-vtuber-comments/README.md)
- [Frontend](../bitnp-raspberrygirl-vtuber-frontend/README.md)
