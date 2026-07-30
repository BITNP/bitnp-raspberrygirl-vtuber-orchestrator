from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
WORKSPACE: Final = ROOT.parent
REPOSITORIES: Final = (
    "bitnp-raspberrygirl-vtuber-mic",
    "bitnp-raspberrygirl-vtuber-comments",
    "bitnp-raspberrygirl-vtuber-orchestrator",
    "bitnp-raspberrygirl-vtuber-sound",
    "bitnp-raspberrygirl-vtuber-frontend",
)
DOCUMENTS: Final = (
    "docs/quickstart.en.md",
    "docs/quickstart.zh-CN.md",
    "docs/deployment.en.md",
    "docs/deployment.zh-CN.md",
    "docs/protocol.en.md",
    "docs/protocol.zh-CN.md",
    "docs/architecture.en.md",
    "docs/architecture.zh-CN.md",
    "docs/testing.en.md",
    "docs/testing.zh-CN.md",
)


def test_documentation_tree_and_protocol_ownership_contract() -> None:
    # Given: the five independent repositories in their sibling workspace.
    # When: their documentation trees and protocol references are inspected.
    for repository_name in REPOSITORIES:
        repository = WORKSPACE / repository_name
        readme = (repository / "README.md").read_text(encoding="utf-8")
        assert all((repository / document).is_file() for document in DOCUMENTS)
        assert "docs/quickstart.en.md" in readme
        assert "docs/quickstart.zh-CN.md" in readme

        protocol = (repository / "docs/protocol.en.md").read_text(encoding="utf-8")
        if repository == ROOT:
            testing = (repository / "docs/testing.en.md").read_text(encoding="utf-8")
            assert "schemas/protocol/envelope.schema.json" in protocol
            assert "schemas/protocol/event-data.schema.json" in protocol
            assert "scripts/verify_protocol_schema.py" in protocol
            assert "--sibling-root" in testing
            assert "--frontend-path" in testing
            continue

        assert "ORCHESTRATOR_REPO" in protocol
        assert "schemas/protocol/envelope.schema.json" in protocol
        assert "schemas/protocol/event-data.schema.json" in protocol
        assert not (repository / "schemas").exists()
        assert not tuple(repository.rglob("*.schema.json"))

        # Then: frontend changes remain limited to its control protocol surface.
    frontend = WORKSPACE / "bitnp-raspberrygirl-vtuber-frontend"
    status = subprocess.run(
        ["/usr/bin/git", "status", "--porcelain"],
        cwd=frontend,
        check=False,
        text=True,
        capture_output=True,
    )
    assert status.returncode == 0, status.stderr
    changed_paths = tuple(line[3:] for line in status.stdout.splitlines())
    assert all(
            path in {
                ".gitignore",
                "README.md",
                "scripts/vtuber_control_client.gd",
                "raspberry_girl.tscn",
                "tests/protocol_smoke.gd",
                "tests/fixtures/vtuber_control_commands.json",
                "tests/fixtures/vtuber_control_invalid.json",
            }
            or path.startswith("docs/")
        for path in changed_paths
    )
