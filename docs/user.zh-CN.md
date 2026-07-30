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

开发和测试默认使用 `ORCHESTRATOR_LLM_PROVIDER=mock`，不需要凭据、GPU、外部服务、真实音频设备或 Godot。

```bash
cd bitnp-raspberrygirl-vtuber-orchestrator
uv sync --locked
uv run pytest
```

跨模块契约、静态检查和部署说明由[开发者文档](developer.zh-CN.md)统一维护。各模块的本地命令在其自己的用户文档中维护。

## 使用指南

### 前端形象与演示控制

Orchestrator 向 Frontend 发送有限的动作、表情、场景和演示命令，用同一个智能体覆盖讲解、宣讲、直播和观众互动等场景。LLM 输出只是候选提案，真正执行前会经过 typed command、能力 allowlist、当前状态和前置条件校验。

模块专属操作请阅读各模块仓库内的用户文档。
