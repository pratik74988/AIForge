from __future__ import annotations

from pathlib import Path

from .knowledge import KnowledgeItem


class MarkdownKnowledgeLoader:
    """
    Loads Markdown documents recursively from a Knowledge Base directory.

    Each Markdown file becomes one KnowledgeItem.
    """

    def __init__(self, knowledge_base_path: str | Path) -> None:
        self._path = Path(knowledge_base_path)

    def load(self) -> list[KnowledgeItem]:
        self._validate_path()

        items: list[KnowledgeItem] = []

        for path in sorted(self._path.rglob("*.md")):
            content = path.read_text(encoding="utf-8").strip()

            if not content:
                continue

            items.append(
                KnowledgeItem(
                    content=content,
                    metadata={
                        "source": str(path),
                        "filename": path.name,
                        "relative_path": path.relative_to(self._path).as_posix(),
                    },
                )
            )

        return items

    def _validate_path(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(
                f"Knowledge Base directory does not exist: {self._path}"
            )

        if not self._path.is_dir():
            raise NotADirectoryError(
                f"Knowledge Base path is not a directory: {self._path}"
            )