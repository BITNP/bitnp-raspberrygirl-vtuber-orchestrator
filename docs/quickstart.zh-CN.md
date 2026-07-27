# 快速开始

使用 Python 3.12 或更高版本。默认的 `ORCHESTRATOR_LLM_PROVIDER=mock` 路径不需要凭据、GPU、外部服务、RTP 硬件或 Godot。

```bash
uv sync --locked
uv run pytest
uv run basedpyright
uv run ruff check src tests
```

本地契约检查可运行 `python scripts/verify_protocol_schema.py` 和 `python scripts/verify_topology.py --sibling-root ..`。
