# 本地知识库与 MCP

知识库为 Brain 提供本地事实；MCP 为 Brain 提供静态允许的外部工具。两者都由 Orchestrator 管理，其他端不得直接调用。知识库检索在首次 Brain 请求前执行；MCP 只接受 Brain 的可信 intent 映射，一轮至多执行一个工具，再调用 Brain 生成最终回复。

## 配置离线知识库

仓库提供从 `bitnp-website` 提取的中文语料 `knowledge/bitnp-website/`，涵盖协会、部门、服务和校园导航；配置方法、来源和时效说明见[网站知识库快照](bitnp-website-knowledge.zh-CN.md)。

在部署环境中设置：

```dotenv
ORCHESTRATOR_KNOWLEDGE_DIR=/srv/raspberrygirl/knowledge
ORCHESTRATOR_KNOWLEDGE_TOP_K=4
```

目录内放置 UTF-8 `.md`、`.txt` 或有效的 `.json` 文件，可以有子目录。没有设置目录时知识库禁用；空目录可用，但不会产生摘录。目录不存在、文件损坏或超过限制会阻止启动，不会静默使用测试数据。

服务在接收连接之前一次性读取语料，通过 LlamaIndex Document 和 SentenceSplitter 分块，再构建中文 BM25 关键词索引。所有会话共享这个不可变快照，新增、修改、删除文件均须重启服务。不会写回文件或持久化索引。语料版本由相对路径和实际读取的内容计算，索引版本还包含分块及排序算法版本。

检索不使用 MockEmbedding，也不下载模型或调用 embedding 服务。中文按单字和连续双字索引，英文和数字按词索引，并进行 Unicode NFKC 和大小写归一化。结果按相关性排序，同分按稳定的文件/分块顺序返回；没有共同关键词时返回空结果。它不提供同义词或跨语言语义匹配，需要在语料中写入常用称呼与别名。

限制为最多 256 个语料文件、目录树 4096 个条目、单文件 1 MiB、总计 16 MiB、8192 个分块。符号链接及非普通语料文件被拒绝。默认返回 4 条，`TOP_K` 可设为 1—8；每条进入 Brain 的摘录最多 4000 字符，包含来源和版本。检索限时 5 秒，并通过协作式让出执行权响应取消，不使用后台检索线程。

## 配置 MCP

复制 `samples/mcp.example.json` 到受保护的本地配置目录，根据实际 MCP 服务修改 server、tool、参数 schema 和中文 description：

```dotenv
ORCHESTRATOR_MCP_CONFIG=/etc/raspberrygirl/mcp.json
CATALOG_MCP_TOKEN=<由部署环境提供>
```

未设置配置路径时不开放外部工具。配置文件最多 64 KiB，版本必须为 1，最多 16 个 server、32 个 tool。未知字段、重复映射、缺失凭据、无效 schema 或不存在的 server 会使启动失败。不要把真实地址、凭据或部署配置提交到仓库。

server 的 `url` 使用 Streamable HTTP 端点；远端必须 HTTPS，本机 `127.0.0.1`、`localhost` 和 `::1` 可以使用 HTTP。禁止 URL 内嵌凭据、查询参数、重定向。`token_env` 可省略；配置后对应环境变量必须存在。私有 CA 可通过 server 的 `ca_path` 指定 PEM 文件，不关闭证书验证。

每个 tool 必须明确登记：

| 字段 | 用途 |
| --- | --- |
| `server`、`tool` | 固定服务和工具名称；模型不能更换目标 |
| `intent` | `mcp.` 开头的唯一可信操作标识 |
| `capability` | 能力名称；执行时同时检查此能力与 `mcp:server/tool` |
| `description` | 面向 Brain 的中文用途说明 |
| `arguments_schema` | 完整 JSON Schema；顶层必须是禁止额外字段的 object，可定义嵌套对象、数组和枚举；禁止 `$ref` 等引用 |
| `timeout_ms` | 初始化和调用的总限时，1—30000 毫秒；同时受轮次剩余预算约束 |
| `max_request_bytes` | 参数 JSON 的 UTF-8 字节上限，最多 65536；协议封装仅增加固定有界开销 |
| `max_response_bytes` | 工具响应 JSON-RPC/SSE 的总字节上限，最多 1 MiB；初始化响应另限 16 KiB |

静态登记同时确定会话的初始能力集合；调度器仍在执行前和提交前重新核验当前能力、轮次、版本、取消 epoch 和期限。完整参数验证采用 JSON Schema，不能通过模型增加配置文件之外的工具或字段。

客户端支持 MCP Streamable HTTP 的 JSON 和 SSE 响应，协商协议版本 `2025-03-26`、`2025-06-18` 或 `2025-11-25`。每次调用创建独立连接与 MCP 会话，执行 `initialize` → `notifications/initialized` → `tools/call`，结束时关闭会话。不会动态发现工具、自动重试或恢复 SSE，也不开放 sampling、elicitation、roots、远程 prompts/resources 或 stdio。工具清单和输入 schema 以本地配置为准，服务端描述不能改变权限。

取消/超时会关闭正在读取的 HTTP 响应，尽力发送 `notifications/cancelled`，并删除会话；远端清理最多额外等待 250 毫秒。远端已经提交的副作用无法由客户端撤销，因此有副作用的工具应由服务端自行实现幂等或事务保护。客户端不重试，避免重复执行。

只有成功的工具结果才能作为不可信观察进入临时上下文，最多包含 512 个规范化文本字符及 server/tool、状态、摘要。图片、音频、资源等非文本内容仅形成大小与摘要说明，不抓取资源 URI，也不把二进制数据放进提示词。协议错误、`isError`、超限、过期和取消均不产生成功观察。网络内容不写入本地知识库；涉及工具的最终回复跳过稳定记忆提取。工具返回指令无法直接触发媒体、PPT、其他 MCP 调用等效果。

现有 PPT 继续走 authenticated Frontend 的 `presentation.*.command` / `presentation.result`，不需要把 PPT 注册成外部 MCP 工具。

## 验证与诊断

```bash
uv run pytest tests/test_knowledge_corpus.py tests/test_mcp_configuration.py tests/test_mcp_http.py tests/test_mcp_loopback.py
```

这些测试使用临时语料、可控响应和真实本机 HTTP 服务，无需外部服务或现场语音。`BITNP_LOG_LEVEL=DEBUG` 可查看知识检索及 MCP 结果日志；包含 trace/session/seq/turn、来源、版本、结果摘要和 outcome，不记录 bearer token 或二进制原文。
