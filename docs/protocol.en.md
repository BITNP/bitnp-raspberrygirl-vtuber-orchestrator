# Normative Protocol

This repository is the protocol authority. The normative JSON Schema files are `schemas/protocol/envelope.schema.json` and `schemas/protocol/event-data.schema.json`. Validate them with:

```bash
python scripts/verify_protocol_schema.py
```

The closed envelope requires schema version, event identity, source, time, trace, session, sequence, and typed event data. Services and frontend clients exchange events only through Orchestrator. Keep schemas and fixtures here; consumers must reference this checkout instead of copying them.
