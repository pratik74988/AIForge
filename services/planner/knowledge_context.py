from __future__ import annotations

from .context import KnowledgeContext
from .knowledge import KnowledgeItem


def build_knowledge_context(
    items: list[KnowledgeItem],
) -> KnowledgeContext:
    """
    Convert retrieved KnowledgeItems into the representation consumed
    by PlanningContext.
    """

    return KnowledgeContext(
        items=tuple(item.content for item in items),
        metadata=tuple(
            item.metadata if item.metadata is not None else {}
            for item in items
        ),
    )