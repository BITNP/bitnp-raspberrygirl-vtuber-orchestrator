# Raspberry Girl VTuber Orchestrator

Orchestrator 是 Raspberry Girl 的中心仓库，拥有规范协议、会话状态、调度、LLM/TTS 边界和跨模块契约验证。Mic 负责 VAD、CAM++、端点检测和 ASR，并通过认证 control connection 提交结构化结果；Mic 不向 Orchestrator 发送 RTP。Orchestrator 是唯一 Gate、Brain、状态写者与 effect 派发者，并只向 Sound 输出 TTS RTP。

现场音频由 Orchestrator 的输出 lease 裁决：每个完成的 TTS 响应使用独立 RTP packetizer 和 epoch/SSRC；只有携带精确 turn、segment 与 epoch 的 Sound `finished` 回报才能释放 lease。完成播放不会注销 Mic/Sound 路由，因此下一轮仍需经新的调度授权，旧 RTP 不会恢复。

- [用户文档](docs/user.zh-CN.md)
- [开发者文档](docs/developer.zh-CN.md)

## 模块文档

- [Mic](../bitnp-raspberrygirl-vtuber-mic/README.md)
- [Sound](../bitnp-raspberrygirl-vtuber-sound/README.md)
- [Comments](../bitnp-raspberrygirl-vtuber-comments/README.md)
- [Frontend](../bitnp-raspberrygirl-vtuber-frontend/README.md)
