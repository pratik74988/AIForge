from __future__ import annotations

from pathlib import Path

from .config import PlannerConfig
from .in_memory_retriever import InMemoryKnowledgeRetriever
from .knowledge import EmptyKnowledgeRetriever, KnowledgeRetriever
from .knowledge_loader import MarkdownKnowledgeLoader
from .knowledge_service import KnowledgeService
from .llm_service import LLMService
from .planner import Planner
from .providers.factory import create_litellm_provider


def create_planner(
    config: PlannerConfig | None = None,
    knowledge_retriever: KnowledgeRetriever | None = None,
    knowledge_base_path: str | Path | None = None,
) -> Planner:
    config = config or PlannerConfig()

    provider = create_litellm_provider(config)
    llm_service = LLMService(provider=provider)

    retriever = _create_knowledge_retriever(
        knowledge_retriever=knowledge_retriever,
        knowledge_base_path=knowledge_base_path,
    )

    knowledge_service = KnowledgeService(retriever)

    return Planner(
        llm_service=llm_service,
        knowledge_service=knowledge_service,
    )


def _create_knowledge_retriever(
    *,
    knowledge_retriever: KnowledgeRetriever | None,
    knowledge_base_path: str | Path | None,
) -> KnowledgeRetriever:
    if knowledge_retriever is not None:
        return knowledge_retriever

    if knowledge_base_path is not None:
        loader = MarkdownKnowledgeLoader(knowledge_base_path)
        items = loader.load()
        return InMemoryKnowledgeRetriever(items)

    return EmptyKnowledgeRetriever()