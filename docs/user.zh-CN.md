# 树莓娘用户文档

树莓娘（Raspberry Girl）是由北京理工大学网络开拓者协会自主设计的虚拟形象，也是北京理工大学网络开拓者协会的官方吉祥物。本项目让树莓娘成为面向公开讲解、虚拟主播和现场产品介绍的单一多模态 AI 角色系统，把听、说、读、推理、记忆、预设动作、演示控制和观众评论接入到一个由 Orchestrator 统一调度的体验中。

## 项目功能

- 语音输入：Mic 采集 16 kHz 单声道音频，在本地完成端点检测与 ASR，并仅在认证 control connection 上将结构化 ASR final 交给 Orchestrator。
- 智能回复：Orchestrator 通过单一 LLM Brain Pipeline 处理语音与评论，调用 LLM、TTS provider，并维护会话、轮次、任务和取消状态。
- 语音输出：Sound 接收 Orchestrator 生成的 L16 RTP 音频并播放。
- 观众输入：Comments 将观众评论规范化为 `audience.input` 事件提交给 Orchestrator。
- 虚拟形象控制：Frontend 接收 Orchestrator 的字幕、动作、场景和演示控制命令；当前动作包括 `hello`、`act_cute` 和 `emphasis`，尚未开放 expression。
- 演示协议：Orchestrator 已实现受控 deck 加载、播放、翻页及回执流程；当前 Frontend 尚无真实 deck 渲染器，会以 `presentation_unavailable` 拒绝合法命令。
- 自适应交互：Orchestrator 根据会话状态、输入来源、可用能力和用户意图选择当前行为，不把产品拆成固定的三种模式。

## 快速开始

开发和测试默认使用 `ORCHESTRATOR_LLM_PROVIDER=mock`，不需要凭据、GPU、外部服务、真实音频设备或 Godot。`.env.example` 展示的是现场语音链路的生产配置形状；应用不会自行读取 `.env`。

```bash
cd bitnp-raspberrygirl-vtuber-orchestrator
uv sync --locked
uv run pytest
```

本地加载环境文件时使用 `uv run --env-file .env orchestrator-transport`。systemd 部署通过 `EnvironmentFile` 注入。完整的安全部署、启动顺序和验收步骤见[部署文档](../deploy/README.md)；受控局域网的认证明文联调见[联调指南](local-loopback.zh-CN.md)。

真实 LLM 部署必须配置 OpenAI-compatible Chat Completions endpoint、模型和所需凭据，并通过 `ORCHESTRATOR_LLM_REASONING_DIALECT` 明确选择 `deepseek`、`openai` 或 `none` 请求方言。`none` 不发送思考参数。Brain、记忆提取与上下文压缩默认均关闭思考；Brain 与 maintenance 可分别指定模型，未指定时使用 `ORCHESTRATOR_LLM_MODEL`。

### LLM 生成参数

以下后缀加上 `ORCHESTRATOR_LLM_` 构成全局环境变量；也可使用
`ORCHESTRATOR_LLM_BRAIN_` 或 `ORCHESTRATOR_LLM_MAINTENANCE_` 前缀分别覆盖。
优先级是任务专属非空值 → 全局非空值 → 原有请求默认值。空白表示继承；
`omit` 表示不发送该参数，不等于数值 0，也不等于关闭思考。
配置在启动时读取，修改后重启。非法数值、非有限数值和未知枚举在启动时拒绝。

| 后缀 | 可配置值 | 未配置时 |
| --- | --- | --- |
| `TEMPERATURE` | 0～2，或 `omit` | Brain 0.2，维护 0 |
| `TOP_P` | 0～1，或 `omit` | 不发送 |
| `FREQUENCY_PENALTY` | -2～2，或 `omit` | 不发送 |
| `PRESENCE_PENALTY` | -2～2，或 `omit` | 不发送 |
| `REASONING` | `enabled` / `disabled` / `omit` | `disabled` |
| `REASONING_EFFORT` | `minimal` / `low` / `medium` / `high` / `xhigh` | openai 方言且 `REASONING=enabled` 时默认 `medium`；disabled 固定发送 `none` |
| `MAX_COMPLETION_TOKENS` | 正整数 | Brain 8192，维护 4096 |
| `TOKEN_PARAMETER` | `auto` / `max_tokens` / `max_completion_tokens` | `auto`：openai 使用后者，deepseek/none 使用前者 |
| `REASONING_DIALECT` | `deepseek` / `openai` / `none` | 任务专属值继承必填的全局方言 |

例如，调整 Brain 温度与预算，同时保持维护调用保守：

```dotenv
ORCHESTRATOR_LLM_REASONING_DIALECT=deepseek
ORCHESTRATOR_LLM_BRAIN_TEMPERATURE=0.6
ORCHESTRATOR_LLM_BRAIN_REASONING=disabled
ORCHESTRATOR_LLM_BRAIN_MAX_COMPLETION_TOKENS=4096
ORCHESTRATOR_LLM_MAINTENANCE_TEMPERATURE=0
ORCHESTRATOR_LLM_MAINTENANCE_REASONING=disabled
ORCHESTRATOR_LLM_MAINTENANCE_MAX_COMPLETION_TOKENS=2048
```

思考参数没有统一协议：`deepseek` 方言发送 `thinking.type`，`openai` 方言发送
`reasoning_effort`（关闭时为 `none`）。如果服务不支持思考控制，选择 `none` 方言
或 `REASONING=omit`；这只省略参数，不能保证服务端停止思考。不支持温度或惩罚项
的模型可将相应参数设为 `omit`。不同维护模型可单独指定方言和预算字段名。
枚举和数值范围表示客户端接受的配置，实际支持范围仍取决于服务和模型；
服务返回错误时保留既有失败处理，不自动删参数重试。通常先调整温度或 top_p 中的一项。

输出预算不会省略，也不会扩大调度器的 deadline、响应长度或效果权限。
推理模型可能将思考 token 计入输出预算，增加预算并不保证在原有时限内完成。
Orchestrator 发送标准 `system`/`user` messages；Chat Template 和 reasoning parser 由模型服务端配置。结构化响应使用 `response_format={"type":"json_object"}`，传入的本地 schema 不会作为 provider 的 JSON Schema 发送，最终安全边界仍是本地严格提案校验。独立 `reasoning_content` 不会进入 speech；思考开关不负责清理 `speech`
字段中混入的思考文本；生成参数配置并不替代播报内容校验。
实际发送的生成参数记录在 `DEBUG` 的 `llm_generation_parameters` 日志中。
DEBUG 还会保留完整的文本提示和 LLM HTTP 响应，包括 provider 返回的
`reasoning_content`；独立的 `reasoning_content` 只用于诊断，不进入 TTS 或字幕。
这类日志可能包含完整对话和知识摘录，应按私有运行日志管理。
跨模块契约、静态检查和部署说明由[开发者文档](developer.zh-CN.md)统一维护。各模块的本地命令在其自己的用户文档中维护。

## 使用指南

### 会话、上下文与知识

Mic input、Sound sink 或 Frontend 注册可以创建 session；Comments 与 operator 只能使用已经存在的 session。每个 session 隔离轮次、临时上下文、记忆、说话人资料和播放状态。失去所有 owner、没有活动任务并超过配置 TTL 后，Orchestrator 会清理该 session 的任务、上下文、记忆、资料和存储。

临时上下文只记录已确认输入、最终 Brain 回复和成功工具观察。会话记忆由独立 maintenance 调用提取，经敏感信息、revision、冲突和容量校验后写入；它不是跨 session 的全局用户画像。本地知识库在进程启动时只读加载，文件变化需要重启。知识库和 MCP 的格式、限额及取消语义见[本地知识库与 MCP](knowledge-mcp.zh-CN.md)。

### 前端形象与演示控制

Orchestrator 向 Frontend 发送有限的字幕、动作、场景和演示命令，用同一个智能体覆盖讲解、宣讲、直播和观众互动等场景。LLM 输出只是候选提案，真正执行前会经过 typed command、能力 allowlist、当前状态和前置条件校验。

部署者通过 `ORCHESTRATOR_PPT_DECK_CATALOG` 提供受控 deck ID 列表；未配置时 Brain 看不到演示操作。加载只允许目录中的 ID，翻页页码限制为 1 到 10000，播放不接受参数。这里的 ID 不是文件路径，Frontend 必须对每条命令返回匹配的成功结果，Orchestrator 才会更新演示状态。当前 Frontend 只实现协议拒绝与幂等回执，因此不能把配置了目录视为 PPT 已可播放。

模块专属操作请阅读各模块仓库内的用户文档。
