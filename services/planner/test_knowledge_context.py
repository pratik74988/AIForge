from .knowledge import KnowledgeItem
from .knowledge_context import build_knowledge_context


def test_build_knowledge_context():
    items = [
        KnowledgeItem(
            content="Scale numeric features when appropriate.",
            metadata={
                "source": "sklearn",
                "topic": "scaling",
            },
        ),
        KnowledgeItem(
            content="Use stratified splitting for imbalanced classification.",
            metadata={
                "source": "ml_guidelines",
                "topic": "splitting",
            },
        ),
    ]

    context = build_knowledge_context(items)

    assert context.items == (
        "Scale numeric features when appropriate.",
        "Use stratified splitting for imbalanced classification.",
    )

    assert context.metadata == (
        {
            "source": "sklearn",
            "topic": "scaling",
        },
        {
            "source": "ml_guidelines",
            "topic": "splitting",
        },
    )


def test_build_knowledge_context_handles_missing_metadata():
    items = [
        KnowledgeItem(
            content="Use median imputation for skewed numeric features."
        )
    ]

    context = build_knowledge_context(items)

    assert context.items == (
        "Use median imputation for skewed numeric features.",
    )

    assert context.metadata == ({},)


def test_build_knowledge_context_handles_no_results():
    context = build_knowledge_context([])

    assert context.items == ()
    assert context.metadata == ()
    