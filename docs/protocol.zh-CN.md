# 规范协议

本仓库是协议权威来源。规范 JSON Schema 文件为 `schemas/protocol/envelope.schema.json` 和 `schemas/protocol/event-data.schema.json`。使用以下命令验证：

```bash
python scripts/verify_protocol_schema.py
```

封闭 envelope 必须包含 schema 版本、事件标识、来源、时间、trace、session、sequence 和有类型的事件数据。服务与前端客户端只能经 Orchestrator 交换事件。schema 与 fixture 保留在此处，消费者应引用此检出目录而不是复制它们。
