# 树莓娘 VTuber Orchestrator

树莓娘（Raspberry Girl）是由北京理工大学网络开拓者协会自主设计的虚拟形象，也是北京理工大学网络开拓者协会的官方吉祥物。本项目让树莓娘成为可以听、说、思考和互动的 AI 虚拟角色，适合会议演示、展台讲解、活动主持和虚拟直播。

“树莓娘”是 Raspberry Girl 的唯一中文名称，文档、界面和智能体回复统一使用此名称。

它可以：

- 降低环境噪声，让现场收音更加清晰
- 自动检测用户何时开始和结束说话
- 识别不同的说话人，提供更个性化的互动
- 听懂现场语音，并自然地回答问题
- 接收 Comments 回放的活动评论，与观众输入使用同一回复流程
- 在会话内维护临时上下文和可审计记忆，让交流更加连贯
- 阅读本地资料，根据已有知识进行讲解
- 使用自然语音播报回答，并支持随时打断
- 控制虚拟形象的逐字字幕、动作和场景
- 通过受控协议请求加载、播放和翻页；Frontend 渲染预先准备的版本化 PPT/PPTX 页面
- 按需使用外部工具，扩展查询和处理能力
- 整个技术栈开源，支持自由部署、按需定制和二次开发

本仓库负责协调树莓娘的各项功能，让语音、评论、回答、播报、虚拟形象和演示内容顺畅配合。

当前可直接运行的现场链路是 Mic 本地 VAD/ASR → Orchestrator Brain/TTS → Sound。Comments 当前提供 JSONL 回放入口；Frontend 已实现字幕 timeline、口型、独立无声动作和本地文稿页面渲染。

## 进一步阅读

- [用户文档](docs/user.zh-CN.md)
- [开发者文档](docs/developer.zh-CN.md)

## 模块文档

- [Mic](../bitnp-raspberrygirl-vtuber-mic/README.md)
- [Sound](../bitnp-raspberrygirl-vtuber-sound/README.md)
- [Comments](../bitnp-raspberrygirl-vtuber-comments/README.md)
- [Frontend](../bitnp-raspberrygirl-vtuber-frontend/README.md)

知识库与外部工具的部署配置见 [本地知识库与 MCP](docs/knowledge-mcp.zh-CN.md)。
