from pathlib import Path
from unittest.mock import Mock, patch

from . import bootstrap
from .config import PlannerConfig
from .in_memory_retriever import InMemoryKnowledgeRetriever
from .knowledge import EmptyKnowledgeRetriever
from .knowledge_service import KnowledgeService
from .llm_service import LLMService
from .planner import Planner


def test_create_planner_uses_configured_knowledge_retriever() -> None:
    config = PlannerConfig(model="test-model")
    retriever = Mock()

    with patch.object(
        bootstrap,
        "create_litellm_provider",
    ) as create_provider:
        planner = bootstrap.create_planner(
            config=config,
            knowledge_retriever=retriever,
        )

    create_provider.assert_called_once_with(config)

    assert isinstance(planner, Planner)
    assert isinstance(planner._llm_service, LLMService)
    assert isinstance(planner._knowledge_service, KnowledgeService)
    assert planner._knowledge_service._retriever is retriever


def test_create_planner_uses_empty_retriever_by_default() -> None:
    config = PlannerConfig(model="test-model")

    with patch.object(
        bootstrap,
        "create_litellm_provider",
    ):
        planner = bootstrap.create_planner(config=config)

    retriever = planner._knowledge_service._retriever

    assert isinstance(retriever, EmptyKnowledgeRetriever)


def test_create_planner_loads_knowledge_base(
    tmp_path: Path,
) -> None:
    knowledge_base = tmp_path / "knowledge-base"
    preprocessing = knowledge_base / "preprocessing"

    preprocessing.mkdir(parents=True)

    (preprocessing / "scaling.md").write_text(
        """
# Feature Scaling

Standard scaling is commonly useful for logistic regression.
""".strip(),
        encoding="utf-8",
    )

    config = PlannerConfig(model="test-model")

    with patch.object(
        bootstrap,
        "create_litellm_provider",
    ):
        planner = bootstrap.create_planner(
            config=config,
            knowledge_base_path=knowledge_base,
        )

    retriever = planner._knowledge_service._retriever

    assert isinstance(
        retriever,
        InMemoryKnowledgeRetriever,
    )

    results = retriever.retrieve(
        "logistic regression scaling"
    )

    assert len(results) == 1
    assert results[0].metadata["filename"] == "scaling.md"


def test_explicit_retriever_takes_precedence_over_path(
    tmp_path: Path,
) -> None:
    knowledge_base = tmp_path / "knowledge-base"
    knowledge_base.mkdir()

    (knowledge_base / "document.md").write_text(
        "Knowledge from filesystem",
        encoding="utf-8",
    )

    explicit_retriever = Mock()
    config = PlannerConfig(model="test-model")

    with patch.object(
        bootstrap,
        "create_litellm_provider",
    ):
        planner = bootstrap.create_planner(
            config=config,
            knowledge_retriever=explicit_retriever,
            knowledge_base_path=knowledge_base,
        )

    retriever = planner._knowledge_service._retriever

    assert retriever is explicit_retriever