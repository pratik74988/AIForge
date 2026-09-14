"""
Core Planner orchestration for AIForge.
"""

from __future__ import annotations

from dataclasses import replace

from .context import PlanningContext
from .knowledge_service import KnowledgeService
from .llm_service import LLMService
from .models.planner_result import PlannerResult


class Planner:
    """
    Coordinates the AIForge planning workflow.

    Flow:

        PlanningContext
            -> KnowledgeService
            -> LLMService
            -> PlannerResult
    """

    def __init__(
        self,
        llm_service: LLMService,
        knowledge_service: KnowledgeService | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._knowledge_service = knowledge_service

    def plan(self, context: PlanningContext) -> PlannerResult:
        if self._knowledge_service is not None:
            knowledge_context = self._knowledge_service.retrieve(context)

            context = replace(
                context,
                knowledge_context=knowledge_context,
            )

        return self._llm_service.generate(context)