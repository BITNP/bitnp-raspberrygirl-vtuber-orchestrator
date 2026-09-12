# 阿里云百炼 CosyVoice HTTP API 调研

调研日期：2026-09-12

范围：仅查阅阿里云百炼官方文档，聚焦 `cosyvoice-v3.5-flash` 的声音复刻与非实时流式语音合成。

## 结论

当前日志中的请求地址不符合百炼 CosyVoice HTTP API。官方合成端点是：

```text
POST https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer
```

不能把该地址当作 OpenAI SDK 的 `base_url`，再由 SDK 追加 `/audio/speech`。日志中的
`.../SpeechSynthesizer/audio/speech` 并不是官方端点。[1]

百炼公开的 CosyVoice HTTP 协议也不支持在每一次合成请求中直接提交 `ref_audio` 和
`ref_text`。正确流程分为两步：先用公网可访问的参考音频 URL 创建一个绑定到
`cosyvoice-v3.5-flash` 的自定义音色，再将返回的 `voice_id` 作为合成请求的
`input.voice`。[2][3]

## 鉴权与地域

`cosyvoice-v3.5-flash` 的上述非实时合成能力仅在华北 2（北京）地域可用。请求使用业务空间专属域名，
其中 `{WorkspaceId}` 替换为真实业务空间 ID；API Key 也必须属于北京地域。[1][4]

两类请求均使用以下请求头：[1][2]

```http
Authorization: Bearer <DASHSCOPE_API_KEY>
Content-Type: application/json
```

流式合成另加：[1]

```http
X-DashScope-SSE: enable
```

## 第一步：为 CosyVoice 创建复刻音色

端点：[2]

```text
POST https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/tts/customization
```

适用于本项目的最小请求如下：

```json
{
  "model": "voice-enrollment",
  "input": {
    "action": "create_voice",
    "target_model": "cosyvoice-v3.5-flash",
    "prefix": "raspberry",
    "url": "https://example.invalid/reference.wav",
    "language_hints": ["zh"]
  }
}
```

关键约束：[2][3]

- `model` 固定为 `voice-enrollment`，`action` 固定为 `create_voice`。
- `target_model` 必须与后续合成所用模型完全一致，否则合成失败；这里应为
  `cosyvoice-v3.5-flash`。
- `prefix` 仅允许数字和英文字母，最长 10 个字符。
- CosyVoice 使用 `input.url`，且 URL 必须能从公网直接访问。
- CosyVoice 这条复刻协议没有参考文本字段；文档中的 `text` 只适用于
  `qwen-voice-enrollment`，不适用于 CosyVoice。
- Data URL（`data:audio/...;base64,...`）只在文档的 Qwen-TTS
  `qwen-voice-enrollment` 分支中受支持；CosyVoice `voice-enrollment` 分支只接受公网音频 URL。
- `language_hints` 当前只处理数组第一个元素；`cosyvoice-v3.5-flash` 支持 `zh`、`en`、
  `fr`、`de`、`ja`、`ko`、`ru`、`pt`、`th`、`id`、`vi`，默认 `zh`。
- 可选 `max_prompt_audio_length` 范围为 3.0–30.0 秒，默认 10.0 秒；可选
  `enable_preprocess` 默认 `false`。

成功响应中的音色标识位于 `output.voice_id`：[2]

```json
{
  "output": {
    "voice_id": "cosyvoice-v3.5-flash-raspberry-xxxxxx"
  },
  "usage": {"count": 1},
  "request_id": "xxxx-xxxx-xxxx"
}
```

参考音频应为 16-bit WAV、MP3 或 M4A，推荐 10–20 秒、最长 60 秒、文件不超过
10 MB、采样率至少 16 kHz。单/双声道均可，但双声道只处理首声道；音频须至少包含
5 秒连续清晰朗读，避免背景音乐、环境噪音和其他人声。[3]

若调用方只有本地文件或 Data URL，必须先把音频转换为 CosyVoice 可访问的 URL。百炼提供一种
官方临时上传流程：先获取上传凭证并上传文件，得到有效期 48 小时的 `oss://` URL；使用它调用
HTTP API 时必须添加 `X-DashScope-OssResourceResolve: enable`。上传文件绑定主账号与模型，
上传时指定的模型必须与后续调用模型一致。[5] 官方明确说明该临时空间不得用于生产、高并发或
压测，生产环境建议使用阿里云 OSS 等稳定存储。[5]

## 第二步：使用音色 ID 流式合成

请求直接发送到 `SpeechSynthesizer` 端点，不使用 OpenAI `/audio/speech` 资源：[1]

```http
POST /api/v1/services/audio/tts/SpeechSynthesizer HTTP/1.1
Host: {WorkspaceId}.cn-beijing.maas.aliyuncs.com
Authorization: Bearer <DASHSCOPE_API_KEY>
Content-Type: application/json
X-DashScope-SSE: enable
```

若下游需要 16 kHz PCM，适用请求体为：[1]

```json
{
  "model": "cosyvoice-v3.5-flash",
  "input": {
    "text": "你好呀，我是树莓娘。",
    "voice": "cosyvoice-v3.5-flash-raspberry-xxxxxx",
    "format": "pcm",
    "sample_rate": 16000
  }
}
```

`format` 可取 `mp3`、`pcm`、`wav`、`opus`，默认 `mp3`；`sample_rate` 可取
8000、16000、22050（默认）、24000、44100、48000 Hz。[1] 因此，应显式请求
`format: "pcm"` 和 `sample_rate: 16000`，不要依赖默认值。官方 HTTP API 页面只将
`pcm` 定义为音频编码格式，并未在该页面进一步声明位深、字节序或声道数；实现不应从这份文档
额外推断这些属性。

## SSE 返回与音频数据

设置 `X-DashScope-SSE: enable` 后，服务端用 Server-Sent Events（SSE）逐段返回 JSON
结果。[1][4] 每个 JSON 结果包含 `request_id`、`output` 和 `usage`。流式事件的
`output.type` 取值为：[1]

- `sentence-begin`：句子开始，包含分句文本。
- `sentence-synthesis`：一个音频数据块；同一句会有多个该事件，客户端必须按接收顺序追加。
- `sentence-end`：句子结束，包含句子信息与累计计费字符数。

流式音频位于 `output.audio.data`，内容为 Base64 编码；客户端应逐事件解码并顺序拼接。
处理中 `finish_reason` 的文档取值为 `null`（官方 JSON 示例写作字符串 `"null"`），自然结束为
`stop`。实现宜以 `output.type == "sentence-synthesis"` 判断音频块，并只以
`finish_reason == "stop"` 判断自然结束，避免依赖中间态究竟编码为 JSON `null` 还是字符串。
最终结果还包含完整音频文件的
`output.audio.url`，该 URL 有效期 24 小时，以及 `id` 和 `expires_at`。[1]

文档明确指出 `sentence-synthesis` 事件与音频数据块一一对应，不会错位。[1] 它没有承诺
每个 Base64 块恰好对应固定时长的 PCM 帧；如项目需要 20 ms 帧，应在解码后自行缓存并按
640 字节（16 kHz、单声道、16-bit 这一项目内部媒体约定）切帧，而不应把 SSE 事件边界当作
RTP 帧边界。

## 对当前错误的直接解释

日志展示了两个独立的不兼容点：

1. OpenAI SDK 把 `/audio/speech` 追加到了百炼原生 `SpeechSynthesizer` 地址，产生了
   不存在的 `.../SpeechSynthesizer/audio/speech` URL。
2. 约 210 万字符的 WAV Data URL 被作为 `ref_audio` 随合成请求发送；官方 CosyVoice
   复刻接口要求的是公网可访问的 `input.url`，而合成接口只接受预先创建的 `voice_id`，
   不接受 `ref_audio`/`ref_text`。

因此，`InvalidParameter: url error` 与协议不匹配一致。适配器应走百炼原生 HTTP + SSE
协议，并将“参考音频注册为音色”与“使用音色合成”建模为两个步骤。若部署仍只持有本地文件或
Data URL，则在不新增一个可由百炼公网访问的受控上传位置之前，无法调用 CosyVoice 的该复刻接口。

## 官方来源

1. [非实时语音合成 Qwen-Audio-TTS/CosyVoice HTTP API 参考](https://docs.bailian.console.aliyun.com/zh/model-studio/cosyvoice-tts-http-api)
2. [声音复刻 HTTP API 参考](https://docs.bailian.console.aliyun.com/zh/model-studio/voice-clone-design-http-api)
3. [声音复刻用户指南](https://docs.bailian.console.aliyun.com/zh/model-studio/voice-cloning-user-guide)
4. [非实时语音合成用户指南](https://docs.bailian.console.aliyun.com/zh/model-studio/non-realtime-tts-user-guide)
5. [上传本地文件获取临时 URL](https://docs.bailian.console.aliyun.com/zh/model-studio/get-temporary-file-url)
