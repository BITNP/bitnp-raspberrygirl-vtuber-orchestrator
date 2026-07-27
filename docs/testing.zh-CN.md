# 测试

在本检出目录中运行本地检查：

```bash
uv run pytest
uv run basedpyright
uv run ruff check src tests
python scripts/verify_protocol_schema.py
python scripts/verify_topology.py --sibling-root ..
python scripts/verify_vtuber_contract.py --frontend-path ../bitnp-raspberrygirl-vtuber-frontend
```

仅在所有同级路径都存在时使用 `bash scripts/verify_workspace.sh`。真实 adapter smoke test 是显式可选项，设置 `BITNP_REAL_ADAPTER_FAKE_LOCAL=1` 后运行。
