from .in_memory_retriever import InMemoryKnowledgeRetriever
from .knowledge import KnowledgeItem


def make_retriever() -> InMemoryKnowledgeRetriever:
    return InMemoryKnowledgeRetriever(
        [
            KnowledgeItem(
                content=(
                    "Use stratified splitting for "
                    "classification datasets."
                ),
                metadata={"topic": "splitting"},
            ),
            KnowledgeItem(
                content=(
                    "Scale numeric features before "
                    "logistic regression."
                ),
                metadata={"topic": "scaling"},
            ),
            KnowledgeItem(
                content=(
                    "Use SMOTE carefully for "
                    "imbalanced classification."
                ),
                metadata={"topic": "imbalance"},
            ),
        ]
    )


def test_retriever_returns_matching_items():
    retriever = make_retriever()

    results = retriever.retrieve(
        "imbalanced classification"
    )

    assert len(results) == 2

    assert results[0].content == (
        "Use SMOTE carefully for imbalanced classification."
    )

    assert results[1].content == (
        "Use stratified splitting for classification datasets."
    )


def test_retriever_ranks_by_match_count():
    retriever = make_retriever()

    results = retriever.retrieve(
        "classification splitting"
    )

    assert results[0].content == (
        "Use stratified splitting for classification datasets."
    )


def test_retriever_returns_empty_for_no_match():
    retriever = make_retriever()

    results = retriever.retrieve(
        "random forest hyperparameters"
    )

    assert results == []


def test_retriever_returns_empty_for_empty_query():
    retriever = make_retriever()

    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_retriever_is_case_insensitive():
    retriever = make_retriever()

    results = retriever.retrieve(
        "IMBALANCED CLASSIFICATION"
    )

    assert results


def test_retriever_preserves_original_metadata():
    retriever = make_retriever()

    results = retriever.retrieve(
        "logistic regression scaling"
    )

    assert results[0].metadata == {
        "topic": "scaling",
    }