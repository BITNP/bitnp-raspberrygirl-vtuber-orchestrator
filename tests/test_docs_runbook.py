
from __future__ import annotations

from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]


def test_orchestrator_docs_name_local_canonical_contract_and_explicit_paths() -> None:
    # Given: Orchestrator-owned architecture and runbook documentation.


    architecture = (ROOT / "docs/architecture.md").read_text(encoding="utf-8")

    runbook = (ROOT / "docs/runbook.md").read_text(encoding="utf-8")

    docs = architecture + runbook

    # When: canonical protocol and independent command surfaces are inspected.

    required = (
        "schemas/protocol/envelope.schema.json",
        "schemas/protocol/event-data.schema.json",
        "media.stream.command",
        "RTP",
        "page",
        "--frontend-path",
        "--sibling-root",
    )

    # Then: all root-owned contract material points to this checkout.

    assert all(term in docs for term in required)
