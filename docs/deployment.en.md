# Deployment

Keep `ORCHESTRATOR_LLM_PROVIDER=mock` for credential free development. For OpenAI compatible ASR, LLM, or TTS, set the matching `ORCHESTRATOR_ASR_*`, `ORCHESTRATOR_LLM_*`, and `ORCHESTRATOR_TTS_*` endpoint, model, and API key placeholders in the deployment environment. vLLM Omni voice cloning is an OpenAI compatible TTS extension. Never commit real keys.
