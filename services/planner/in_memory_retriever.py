from __future__ import annotations

import re

from .knowledge import KnowledgeItem


class InMemoryKnowledgeRetriever:
    """
    Simple keyword-based Knowledge Base retriever.

    Intended for development and testing. It keeps knowledge items
    in memory and ranks them by the number of query terms they contain.
    """

    def __init__(
        self,
        items: list[KnowledgeItem] | None = None,
    ) -> None:
        self._items = list(items or [])

    def retrieve(self, query: str) -> list[KnowledgeItem]:
        if not query.strip():
            return []

        query_terms = self._tokenize(query)

        if not query_terms:
            return []

        scored_items: list[tuple[int, int, KnowledgeItem]] = []

        for index, item in enumerate(self._items):
            content_terms = self._tokenize(item.content)

            score = len(query_terms & content_terms)

            if score > 0:
                scored_items.append(
                    (score, index, item)
                )

        scored_items.sort(
            key=lambda entry: (-entry[0], entry[1])
        )

        return [
            item
            for _, _, item in scored_items
        ]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {
            token.lower()
            for token in re.findall(r"\b\w+\b", text)
            if len(token) > 2
        }