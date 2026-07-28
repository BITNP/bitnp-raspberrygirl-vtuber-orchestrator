# VTuber Orchestrator

The hub for the Raspberry Girl distributed AI VTuber system. It owns canonical protocol schemas and the ASR, LLM, and TTS provider boundary.

`orchestrator-transport` is the hub runtime for authenticated WSS control and UDP RTP forwarding. Mic and Sound connect only to this hub. The transport deployment guide covers its TLS, LAN token, network, and verification requirements.

- [English quickstart](docs/quickstart.en.md)
- [简体中文快速开始](docs/quickstart.zh-CN.md)
- [English transport deployment](docs/deployment.en.md)
- [中文传输部署](docs/deployment.zh-CN.md)
- [Protocol](docs/protocol.en.md)
