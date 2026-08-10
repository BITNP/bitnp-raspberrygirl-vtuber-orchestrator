# Raspberry Girl VTuber Orchestrator

Raspberry Girl 是一个面向会议演示、展台讲解和虚拟直播的多模态 AI 角色系统，目标是让智能体既有真人般自然的互动体验，也有可用于真实现场的可靠边界。项目特色包括：

- 语音与评论共用一个 Brain，保持统一的人格、上下文和决策
- 可打断的低延迟语音交互，新回答就绪前不中断当前播放
- 会话上下文、长期记忆与只读知识库彼此分离
- 内置本地知识检索、受控 MCP、虚拟形象动作和 PPT 演示能力
- LLM 只提出建议，所有工具与舞台效果都经过权限、状态和参数校验
- 有界任务、取消代次和完整关联日志，阻止迟到结果产生副作用

本仓库包含 Raspberry Girl 的中心 Orchestrator：它拥有规范协议、会话状态、任务调度、LLM/TTS 边界和跨模块契约验证，是系统唯一的状态写者与效果派发者。Mic、Sound、Comments 和 Frontend 均通过认证连接与它协作，不直接相互通信。

Mic 负责 VAD、CAM++、端点检测和 ASR，并通过认证 control connection 提交结构化结果；Mic 不向 Orchestrator 发送 RTP。Orchestrator 的单一 Brain Pipeline 是唯一业务决策入口，并只向 Sound 输出 TTS RTP。

现场音频由 Orchestrator 的输出 lease 裁决：每个完成的 TTS 响应使用独立 RTP packetizer 和 epoch/SSRC；只有携带精确 turn、segment 与 epoch 的 Sound `finished` 回报才能释放 lease。完成播放不会注销 Mic/Sound 路由，因此下一轮仍需经新的调度授权，旧 RTP 不会恢复。

生产演示操作只在配置非空 `ORCHESTRATOR_PPT_DECK_CATALOG` 时向 Brain 暴露。目录是最多 32 个受控 deck ID 的逗号分隔列表，不接受路径；`load`、`navigate`、`play` 分别使用固定的 `deck_id`、有界 `page` 和空参数 schema，并且必须由当前 session 的 Frontend 精确回执后才提交状态。

## 进一步阅读

- [用户文档](docs/user.zh-CN.md)
- [开发者文档](docs/developer.zh-CN.md)

## 模块文档

- [Mic](../bitnp-raspberrygirl-vtuber-mic/README.md)
- [Sound](../bitnp-raspberrygirl-vtuber-sound/README.md)
- [Comments](../bitnp-raspberrygirl-vtuber-comments/README.md)
- [Frontend](../bitnp-raspberrygirl-vtuber-frontend/README.md)
