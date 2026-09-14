from __future__ import annotations

from .context import PlanningContext
from .knowledge import KnowledgeRetriever
from .knowledge_context import build_knowledge_context


class KnowledgeService:
    """
    Retrieves relevant Knowledge Base guidance for a planning request.

    The service does not make ML decisions. It only retrieves guidance
    that can later be supplied to the LLM Planner.
    """

    def __init__(self, retriever: KnowledgeRetriever) -> None:
        self._retriever = retriever

    def retrieve(self, context: PlanningContext):
        """
        Retrieve Knowledge Base context for the planning request.
        """

        query = self._build_query(context)

        items = self._retriever.retrieve(query)

        return build_knowledge_context(items)

    @staticmethod
    def _build_query(context: PlanningContext) -> str:
        """
        Build a retrieval query from user intent and known dataset facts.
        """

        return context.user_requirement