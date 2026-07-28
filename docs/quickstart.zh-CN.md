# 快速开始

使用 Python 3.12 或更高版本。默认的 `ORCHESTRATOR_LLM_PROVIDER=mock` 路径不需要凭据、GPU、外部服务、RTP 硬件或 Godot。

```bash
uv sync --locked
uv run pytest
uv run basedpyright
uv run ruff check src tests
```

本地契约检查可运行 `python scripts/verify_protocol_schema.py` 和 `python scripts/verify_topology.py --sibling-root ..`。

真实现场讲解部署应使用 `ORCHESTRATOR_MODE=onsite_explainer`，并在 `.env.example` 中配置 OpenAI 兼容 ASR、LLM 和 `vllm_omni` TTS 设置。运行命令是 `uv run orchestrator-transport`。必须先启动它，再启动 Sound 和 Mic。请参阅[部署指南](deployment.zh-CN.md)和 [systemd 部署包](../deploy/README.md)，frontend 不属于此链路。
