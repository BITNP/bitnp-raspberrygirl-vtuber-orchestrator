# 部署

无凭据开发请保留 `ORCHESTRATOR_LLM_PROVIDER=mock`。使用 OpenAI 兼容的 ASR、LLM 或 TTS 时，在部署环境中设置对应的 `ORCHESTRATOR_ASR_*`、`ORCHESTRATOR_LLM_*` 和 `ORCHESTRATOR_TTS_*` endpoint、model 与 API key 占位配置。vLLM Omni 声音克隆是 OpenAI 兼容的 TTS 扩展。不得提交真实密钥。
