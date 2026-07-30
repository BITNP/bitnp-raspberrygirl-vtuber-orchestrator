# 架构

Orchestrator 是 Mic、Comments、Sound 和前端的中心。只有它负责跨服务路由、交互策略与 provider 决策。Mic 和 Sound 通过 RTP 连接中心，所有控制流量使用规范协议。

Raspberry Girl 是一个单一的多模态 AI 智能体，而不是三个固定产品模式的集合。Orchestrator 根据会话状态、输入来源、可用能力和用户意图选择当前交互策略；前端只执行 Orchestrator 发出的明确命令；Mic、Comments 和 Sound 保持不感知业务策略。Orchestrator 拥有可配置的 OpenAI 兼容 ASR、LLM 与 TTS provider。

## 现场语音交互音频桥接

现场语音交互是由中心拥有的音频替换路径。启用该链路时，Mic 与 Sound 仍通过它们原有的 WSS 控制交换连接 Orchestrator，并保留原有的 RTP 边界，因此两个服务都不需要业务策略特定行为。前端不参与此音频部署。

```text
Mic  -- WSS 源控制 --> Orchestrator <-- WSS 汇控制 -- Sound
Mic  -- UDP L16 RTP --> Orchestrator -- UDP 生成的 L16 RTP --> Sound
                              |
+-> VAD 端点 -> ASR -> turn pipeline -> LLM -> TTS
```

WSS 控制为同一个 session 与 stream 注册 Mic 源和 Sound 汇。Orchestrator 校验并固定已接受的 Mic RTP 路由，然后向 Sound 发送带有生成输出 SSRC 的 `media.stream.command`。在现场语音交互链路中，原始 Mic RTP 不会被转发。生成的音频会在 Sound RTP 边界替换它。

桥接以确定性 VAD 对已接受的 20 ms L16 RTP 帧进行端点检测：任一样本绝对值达到 400 即开始语音，最多保留十帧 20 ms 预滚动，连续三十帧静音（600 ms）结束话语，750 帧（15 s）强制结束。重复包会丢弃，允许一帧乱序，间隙会结束当前话语，断开连接只结束一次。在 provider 边界，网络序 L16 样本会交换字节序，转换为 PCM16LE，并封装为供 ASR 使用的 16 kHz 单声道 PCM WAV。已配置的 OpenAI 兼容 ASR、OpenAI 兼容 LLM 和 vLLM-Omni TTS 通过 turn pipeline 组合，并通过工作线程卸载在 UDP 回调之外运行。TTS 输出必须是 `audio/wav`、16 kHz、单声道、未压缩的 PCM16。桥接会校验 PCM16LE WAV 样本，交换回网络序 L16，填充至完整 RTP 载荷，再以从 Mic SSRC 确定性派生的非零生成 SSRC 进行分包。

provider 或媒体失败不会产生生成输出。空白 ASR 结果也不会产生输出。stream 取消、移除、断开连接和运行时清理都会使待处理的桥接作业失效。路由 generation 会在作业完成时进行门控，因此已取消或已断开的路由会丢弃迟到输出，而不会把它发送到 Sound。

测试覆盖 L16 与 PCM16LE 的字节序转换、固定 WAV 校验、空白 ASR 的无输出行为、Sound 命令与生成数据包的兼容性，以及取消时对过期输出的抑制。配置与运行流程请参阅[部署文档](deployment.zh-CN.md)。
