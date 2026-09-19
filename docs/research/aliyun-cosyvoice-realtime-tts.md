# 阿里云百炼 CosyVoice 实时语音合成接入

调研日期：2026-09-19

范围：仅查阅阿里云百炼官方「实时语音合成」文档（仓库根目录的本地快照
`实时语音合成 - 阿里云百炼.html`），聚焦 CosyVoice 系列的 WebSocket 实时合成与
DashScope Python SDK 用法。

## 结论

`aliyun_cosyvoice` provider 已从百炼非实时 HTTP `SpeechSynthesizer` + SSE 改为官方
**实时语音合成**协议，并通过官方 DashScope Python SDK（`dashscope.audio.tts_v2`）
调用。官方文档规定：通过 DashScope SDK 调用需要安装最新版 SDK；实时语音合成支持流式
输入与输出，音频按数据块实时返回。

与旧实现相比，三处行为发生变化：

1. `ORCHESTRATOR_TTS_ENDPOINT` 现在是实时推理 WebSocket 地址，而不是 HTTP 资源：

   ```text
   wss://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference
   ```

   共享地址为 `wss://dashscope.aliyuncs.com/api-ws/v1/inference`。适配器在构造时校验
   该字段必须是 `ws://` 或 `wss://`，因此把旧的 `.../api/v1/services/audio/tts/SpeechSynthesizer`
   地址直接配置进来会在启动阶段失败，而不是在合成阶段超时。

2. 合成不再手工构造 `model`/`input.text`/`input.voice` JSON，也不再轮询 SSE 事件或
   下载结果 WAV。适配器使用官方示例的调用形态：

   ```python
   import dashscope
   from dashscope.audio.tts_v2 import AudioFormat, ResultCallback, SpeechSynthesizer

   dashscope.api_key = "<api-key>"
   synthesizer = SpeechSynthesizer(
       model="cosyvoice-v3-flash",
       voice="<系统音色或复刻音色 ID>",
       format=AudioFormat.PCM_16000HZ_MONO_16BIT,
       url="wss://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference",
       callback=callback,
   )
   synthesizer.streaming_call("你好呀，我是树莓娘。")
   synthesizer.streaming_complete()
   ```

   SDK 在 `ResultCallback.on_data(data: bytes)` 上按块推送音频；`on_complete`、
   `on_error`、`on_close` 报告任务终态。`streaming_complete` 发送 finish-task 并等待
   剩余音频，`streaming_cancel` 用于提前终止任务。

3. 输出采样率直接按 16 kHz 请求（`AudioFormat.PCM_16000HZ_MONO_16BIT`），替代了旧实现
   中 `_Pcm24khzTo16khzResampler` 的 24 kHz → 16 kHz 线性重采样。

## 官方示例要点

- 模型与音色必须成对匹配：`cosyvoice-v3-flash`/`cosyvoice-v3-plus` 使用 `longanyang`
  等系统音色，`cosyvoice-v2` 使用 `longxiaochun_v2` 等。跨模型使用音色会导致合成失败，
  这与旧 HTTP 协议的限制一致，仍见
  [aliyun-cosyvoice-http-api.md](aliyun-cosyvoice-http-api.md)。
- `cosyvoice-v3.5-plus`/`cosyvoice-v3.5-flash` 只在北京地域可用，且仅支持声音设计/声音
  复刻，没有系统音色；使用前必须先用复刻或设计接口取得音色 ID。
- 复刻与合成是两条独立协议：复刻接口产生 `voice_id`，实时合成只接受该 ID，不接受
  `ref_audio`/`ref_text`。因此适配器的 `stream_pcm16le`/`synthesize` 仍然忽略参考字段。
- 官方提醒回调运行在 WebSocket 线程上，回调内的阻塞逻辑会影响数据接收，建议把音频写入
  独立缓冲区后再处理。适配器据此把回调实现为纯入队操作，由消费线程完成帧化与 RTP 输出。
- SDK 从模块级全局变量 `dashscope.api_key` 读取凭证（构造 `SpeechSynthesizer` 时即读取），
  因此适配器在打开任务前安装该值。凭证不写入日志、请求载荷或适配器状态。
- 实时 SDK 自行管理 WebSocket TLS 配置，不接受 Orchestrator 的
  `ORCHESTRATOR_TLS_CA_PATH`。百炼公网端点的证书由系统信任链验证，因此该 provider 的
  `ca_path` 字段被接受但不生效。

## 适配器取舍

- `stream_pcm16le` 在独立线程上驱动 `streaming_call` + `streaming_complete`，消费线程从
  无界事件队列读取 PCM 块并逐个 `yield`，因此首包延迟只受 WebSocket 建连与首块合成影响。
- 缓冲模式（`capability=final_only`）复用同一条实时链路，把 16 kHz PCM 包进标准 WAV
  容器后再交给既有媒体路径，用于兼容既有 `l16_from_wav` 与重采样分支。这样两种模式共享
  同一套取消与超时行为，不再需要单独的可取消 HTTP 请求。
- 取消绑定只向事件队列写入终止标记，不阻塞取消方线程；真正的
  `streaming_cancel` 在生成器收尾时执行（上限 2 秒），随后关闭 WebSocket 并等待工作
  线程退出（上限 5 秒）。任何路径都不会无限等待 SDK 内部事件。
- `streaming_complete` 的等待上限为 60 秒，用于覆盖服务端在收到 finish-task 后仍需
  合成剩余文本的情况；socket 级 `on_close`/`on_error` 会立即把中断转成类型化错误，
  不会出现“连接已断开但仍等待超时”的静默停顿。
- 每个 PCM 块经 `binary_summary` 记录为 DEBUG 摘要（字节数与十六进制前缀），文本与
  请求形状按完整明文记录；API Key 与音频二进制正文不进入日志。

## SDK 版本与测试

- 依赖：`dashscope>=1.25.2`（`uv.lock` 当前解析为 1.27.6），同时新增
  `websocket-client`、`httpx-sse` 等传递依赖。
- `dashscope` 包在导入时会为与实时语音合成无关的 Assistants API 发出
  `DeprecationWarning`。本仓库的 pytest 配置把 warning 视为错误，因此在
  `pyproject.toml` 中为该第三方导入告警添加了一条窄范围、带说明的
  `filterwarnings` 例外。
- 适配器单测用脚本化 fake 替换 SDK 工厂：先构造真实 `SpeechSynthesizer` 校验
  model/voice/format/url 与 `dashscope.api_key` 的装配，再验证流式分块、任务失败、
  取消与缓冲成 WAV 的行为。测试不建立网络连接。

## 官方来源

1. 阿里云百炼「实时语音合成」文档（本地快照：`实时语音合成 - 阿里云百炼.html`），含
   CosyVoice 快速开始、`SpeechSynthesizer` 对象池与取消说明、FAQ。
2. 实时语音合成 - Qwen-Audio-TTS/CosyVoice API 参考（上述文档「API参考」小节）。
3. [获取 API Key](https://help.aliyun.com/zh/model-studio/get-api-key)。
