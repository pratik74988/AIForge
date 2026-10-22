from unittest.mock import Mock

from .context import PlanningContext
from .knowledge import KnowledgeItem
from .knowledge_service import KnowledgeService


def make_context() -> PlanningContext:
    return PlanningContext(
        user_requirement="Build a churn classification model.",
        dataset_profile={
            "target": {
                "name": "churn",
                "task": "classification",
            }
        },
    )


def test_knowledge_service_retrieves_using_user_requirement():
    retriever = Mock()

    retriever.retrieve.return_value = [
        KnowledgeItem(
            content="Use stratified splitting for classification.",
            metadata={
                "topic": "splitting",
            },
        )
    ]

    service = KnowledgeService(retriever)

    context = make_context()

    result = service.retrieve(context)

    retriever.retrieve.assert_called_once_with(
        "Build a churn classification model."
    )

    assert result.items == (
        "Use stratified splitting for classification.",
    )

    assert result.metadata == (
        {
            "topic": "splitting",
        },
    )


def test_knowledge_service_returns_empty_context_when_no_results():
    retriever = Mock()
    retriever.retrieve.return_value = []

    service = KnowledgeService(retriever)

    result = service.retrieve(make_context())

    assert result.items == ()
    assert result.metadata == ()

    retriever.retrieve.assert_called_once_with(
        "Build a churn classification model."
    )


def test_knowledge_service_does_not_modify_retrieved_content():
    retriever = Mock()

    item = KnowledgeItem(
        content="Logistic regression benefits from appropriate feature scaling.",
        metadata={"source": "ml_guidelines"},
    )

    retriever.retrieve.return_value = [item]

    service = KnowledgeService(retriever)

    result = service.retrieve(make_context())

    assert result.items[0] == item.content
    assert result.metadata[0] == item.metadata