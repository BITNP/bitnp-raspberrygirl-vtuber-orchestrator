# Raspberry Girl 用户文档

Raspberry Girl 是一个面向公开讲解、虚拟主播和现场产品介绍的单一多模态 AI 角色系统。它把听、说、读、推理、记忆、预设动作、演示控制和观众评论接入到一个由 Orchestrator 统一调度的体验中，目标是在演讲、展台和直播场景里提供响应及时、边界清晰、有真人感的 AI 智能体。

## 项目功能

- 语音输入：Mic 采集 16 kHz 单声道音频，通过 RTP 交给 Orchestrator。
- 智能回复：Orchestrator 调用 ASR、LLM、TTS provider，并维护会话、轮次、任务和取消状态。
- 语音输出：Sound 接收 Orchestrator 生成的 L16 RTP 音频并播放。
- 观众输入：Comments 将观众评论规范化为 `audience.input` 事件提交给 Orchestrator。
- 虚拟形象控制：Frontend 接收 Orchestrator 的表情、动作、场景和演示控制命令。
- 演示支持：支持显式的 deck 加载、播放和翻页命令，以及 Frontend 回执。
- 自适应交互：Orchestrator 根据会话状态、输入来源、可用能力和用户意图选择当前行为，不把产品拆成固定的三种模式。

## 快速开始

开发和测试默认使用 mock LLM provider，不需要凭据、GPU、外部服务、真实音频设备或 Godot。

```bash
cd bitnp-raspberrygirl-vtuber-orchestrator
uv sync --locked
uv run pytest
uv run basedpyright
uv run ruff check src tests
python scripts/verify_protocol_schema.py
python scripts/verify_topology.py --sibling-root ..
python scripts/verify_vtuber_contract.py --frontend-path ../bitnp-raspberrygirl-vtuber-frontend
```

每个 Python 模块也可以单独验证：进入 `bitnp-raspberrygirl-vtuber-mic`、`bitnp-raspberrygirl-vtuber-sound` 或 `bitnp-raspberrygirl-vtuber-comments` 后运行 `uv sync --locked && uv run pytest`。Frontend 使用 Godot 4.6，主场景是 `res://raspberry_girl.tscn`。

## 使用指南

### 现场语音交互链路

现场语音交互使用 Orchestrator 拥有的音频替换链路：Mic 把 L16 RTP 发给 Orchestrator，Orchestrator 完成 VAD、ASR、LLM 和 vLLM-Omni TTS，再把生成的 L16 RTP 发给 Sound。Frontend 不参与这个音频部署。

启动顺序固定为：

```bash
uv run orchestrator-transport
uv run sound-receive
uv run mic-stream
```

生产控制面使用 Orchestrator 的 `/control` WSS endpoint，Mic 和 Sound 必须使用同一个 session ID 和 stream ID，并共享可信局域网 token。Mic 与 Sound 不直接通信。

### 前端形象与演示控制

Orchestrator 向 Frontend 发送有限的动作、表情、场景和演示命令，用同一个智能体覆盖讲解、宣讲、直播和观众互动等场景。LLM 输出只是候选提案，真正执行前会经过 typed command、能力 allowlist、当前状态和前置条件校验。

### 模块入口

- Orchestrator: `uv run orchestrator-transport`
- Mic: `uv run mic-stream`，本地单帧诊断为 `uv run mic-capture`
- Sound: `uv run sound-receive`，本地单包诊断为 `uv run sound-play`
- Comments: `uv run comments-replay`，健康检查为 `uv run comments-health`
- Frontend: 用 Godot 4.6 打开仓库并运行 `res://raspberry_girl.tscn`

更多模块专属操作请阅读各模块仓库内的用户文档。
