# 架构

Orchestrator 是 Mic、Comments、Sound 和前端的中心。只有它负责跨服务路由与 provider 决策。Mic 和 Sound 通过 RTP 连接中心，所有控制流量使用规范协议。

只有 Orchestrator 和前端感知模式，并解释 `lecturer`、`virtual_streamer` 和 `onsite_explainer`。Orchestrator 拥有可配置的 OpenAI 兼容 ASR、LLM 与 TTS provider，其他服务保持不感知模式。
