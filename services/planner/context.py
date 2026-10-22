"""
Planning context models for AIForge.

This module defines the structured information supplied to the Planner
before reasoning begins.

The context deliberately separates:
    - user intent
    - Dataset Analyzer facts
    - Knowledge Base context

The LLM will consume this context later, but this module itself contains
no LLM-specific logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KnowledgeContext:
    """
    Information retrieved from the AIForge Knowledge Base.

    Each item represents relevant knowledge that may help the Planner
    make a decision.

    The Planner does not assume a particular vector database or retrieval
    implementation. Those details belong to the Knowledge Base layer.
    """

    items: tuple[str, ...] = ()

    metadata: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class PlanningContext:
    """
    Complete context available to the Planner.

    Args:
        user_requirement:
            Natural-language description of the user's ML objective.

        dataset_profile:
            Dataset Analyzer output containing factual observations
            about the dataset.

        knowledge_context:
            Relevant Knowledge Base information retrieved for this
            planning request.
    """

    user_requirement: str
    dataset_profile: Any
    knowledge_context: KnowledgeContext = field(
        default_factory=KnowledgeContext
    )