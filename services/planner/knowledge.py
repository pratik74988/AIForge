from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class KnowledgeItem:
    """
    A single piece of guidance retrieved from the Knowledge Base.
    """

    content: str
    metadata: dict[str, Any] | None = None


class KnowledgeRetriever(Protocol):
    """
    Interface required by the Planner for Knowledge Base retrieval.
    """

    def retrieve(self, query: str) -> list[KnowledgeItem]:
        ...


class EmptyKnowledgeRetriever:
    """
    Retriever used when no Knowledge Base backend is configured.

    This allows the Planner to operate without making Knowledge Base
    availability a hard dependency.
    """

    def retrieve(self, query: str) -> list[KnowledgeItem]:
        return []