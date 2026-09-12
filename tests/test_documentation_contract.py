
from __future__ import annotations

from pathlib import Path
from re import finditer
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

MINIMUM_DOCUMENTS: Final = ("docs/user.zh-CN.md", "docs/developer.zh-CN.md")
MARKDOWN_LINK_TARGETS: Final = r"\[[^]]+\]\(([^)]+)\)"


def test_documentation_tree_and_protocol_ownership_contract() -> None:
    # Given: the five independent repositories in their sibling workspace.

    # When: their documentation trees and protocol references are inspected.


    for repository_name in REPOSITORIES:
        repository = WORKSPACE / repository_name

        readme = (repository / "README.md").read_text(encoding="utf-8")

        assert all((repository / document).is_file() for document in MINIMUM_DOCUMENTS)

        assert "docs/user.zh-CN.md" in readme

        assert "docs/developer.zh-CN.md" in readme

        assert ".en.md" not in readme

        developer = (repository / "docs/developer.zh-CN.md").read_text(
            encoding="utf-8"
        )

        if repository == ROOT:
            assert "schemas/protocol/envelope.schema.json" in developer

            assert "schemas/protocol/event-data.schema.json" in developer

            assert "scripts/verify_protocol_schema.py" in developer

            assert "--sibling-root" in developer

            assert "--frontend-path" in developer

            continue

        assert "ORCHESTRATOR_REPO" in developer

        assert "schemas/protocol/envelope.schema.json" in developer

        assert "schemas/protocol/event-data.schema.json" in developer

        assert not (repository / "schemas").exists()

        assert not tuple(repository.rglob("*.schema.json"))



def test_documentation_links_resolve_within_the_workspace() -> None:
    # Given: the retained Markdown documentation in each sibling repository.

    # When: relative Markdown links are resolved from their source documents.

    for repository_name in REPOSITORIES:
        repository = WORKSPACE / repository_name
        documents = (repository / "README.md", *(repository / "docs").glob("*.md"))

        for document in documents:
            content = document.read_text(encoding="utf-8")

            for match in finditer(MARKDOWN_LINK_TARGETS, content):
                target = match.group(1)
                path = target.split("#", maxsplit=1)[0]

                if not path or "://" in path:
                    continue

                # Then: every retained relative link identifies an existing file.
                assert (document.parent / path).resolve().is_file(), (
                    f"broken documentation link in {document}: {target}"
                )
