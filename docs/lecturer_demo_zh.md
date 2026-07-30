# 讲稿演示 Demo

本文档说明如何运行一个本地可交付的讲稿演示 Demo。Demo 由 Orchestrator 读取用户编写的讲稿，直接完成语音合成，并按讲稿中的幻灯片、讲解文字、表情、角色动作和场景名称生成规范协议事件。

## 快速开始

在仓库根目录运行：

```bash
python scripts/run_lecturer_demo.py --script samples/lecturer/bitnet_intro_zh.json --evidence .omo/evidence/lecturer-demo.json
```

成功时终端会输出 `LECTURER DEMO PASSED`，并生成 `.omo/evidence/lecturer-demo.json`。该文件记录完整调试日志，包括 `media.stream.command`、`media.stream.state`、`vtuber.caption.command`、`vtuber.expression.command`、`vtuber.action.command` 和 `vtuber.scene.command`。

这个 Demo 是进程内协议演示，不启动真实 WebSocket 服务，不需要 GPU、API key、麦克风、声卡、Bilibili 凭据或可见的 Godot 窗口。

## 讲稿格式

讲稿使用 JSON，由用户编写。顶层字段：

- `title`：讲稿标题。
- `voice`：OpenAI-compatible TTS 提供方使用的声音名称。TTS 仅是 Orchestrator 的提供方能力，不部署独立的 TTS 服务。
- `steps`：讲解步骤数组，至少包含一个步骤。

每个步骤包含：

- `id`：步骤 ID。
- `narration`：讲解文字。Orchestrator 直接合成音频，并生成与 RTP 流起点对齐的字幕提示。
- `slide`：幻灯片翻页指令，包含 `id`、`title` 和正整数 `page`。
- `expression`：角色表情名称；Orchestrator 生成 RTP 定时的 `vtuber.expression.command`。
- `action`：角色动作名称，例如 `explain_point` 或 `point_slide`。动作由之后的 Godot 设计人员映射到具体动画。
- `scene`：前端场景名称，例如 `lecture_slide_focus`。

示例文件是 `samples/lecturer/bitnet_intro_zh.json`。

## 通信协议

Orchestrator 是唯一中心，并拥有 OpenAI-compatible TTS 提供方的直接合成能力。合成完成后，Orchestrator 向 Sound 发送 `media.stream.command`，其中带有 `stream_id`、音频元数据和 RTP 相对 `start_at_ms`；Sound 以 `media.stream.state` 回报该 RTP 流的播放状态。Sound 只消费 RTP 流媒体，不参与合成提供方调用。

前端同步完全由 Orchestrator 的 RTP 定时提示驱动：`vtuber.caption.command`、`vtuber.expression.command`、`vtuber.action.command` 和 `vtuber.scene.command` 共享同一流相对时间；场景提示携带正整数幻灯片页码。前端不需要等待墙钟时间或请求独立语音服务。

每个协议事件都包含 `schema_version`、`event_type`、`event_id`、`source`、`time`、`trace_id`、`session_id`、`seq` 和 `data`。涉及讲解段落的事件还包含 `turn_id` 和 `segment_id`。

## 开发者文档

代码入口：

- 讲稿解析：`orchestrator/src/orchestrator/lecturer_script.py`
- 本地 Demo：`scripts/run_lecturer_demo.py`
- 协议契约：`orchestrator/src/orchestrator/pipeline_contracts.py`
- 示例讲稿：`samples/lecturer/bitnet_intro_zh.json`

调试日志写入 `--evidence` 指定的 JSON 文件。重点查看：

- `events`：按顺序记录所有协议事件。
- `topology.edges`：记录模块边界。
- `topology.peer_edges`：必须为空，表示没有非 Orchestrator 模块之间的直接通信。

建议验证命令：

```bash
(cd orchestrator && uv run pytest tests/test_lecturer_script.py tests/test_lecturer_protocol.py)
python -m unittest tests.test_lecturer_demo_runner tests.test_lecturer_demo_docs
python scripts/verify_topology.py --root .
```

## 必要注释说明

代码中的注释只用于解释公开协议边界或测试的 Given/When/Then 场景。讲稿格式和调试流程以本文档为准；不要在业务代码中加入面向用户的长说明。
