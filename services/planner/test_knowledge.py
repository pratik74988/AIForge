from .knowledge import EmptyKnowledgeRetriever, KnowledgeItem


def test_knowledge_item_stores_content_and_metadata():
    item = KnowledgeItem(
        content="Numeric features may require scaling.",
        metadata={
            "source": "sklearn_guidelines",
            "topic": "preprocessing",
        },
    )

    assert item.content == "Numeric features may require scaling."
    assert item.metadata == {
        "source": "sklearn_guidelines",
        "topic": "preprocessing",
    }


def test_knowledge_item_metadata_is_optional():
    item = KnowledgeItem(
        content="Use stratified splitting for classification."
    )

    assert item.content == (
        "Use stratified splitting for classification."
    )
    assert item.metadata is None


def test_empty_knowledge_retriever_returns_no_items():
    retriever = EmptyKnowledgeRetriever()

    result = retriever.retrieve(
        "How should I preprocess numeric features?"
    )

    assert result == []


def test_empty_knowledge_retriever_does_not_depend_on_query():
    retriever = EmptyKnowledgeRetriever()

    assert retriever.retrieve("query one") == []
    assert retriever.retrieve("query two") == []