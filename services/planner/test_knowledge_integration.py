from pathlib import Path

from .in_memory_retriever import InMemoryKnowledgeRetriever
from .knowledge_loader import MarkdownKnowledgeLoader


def test_knowledge_base_can_be_loaded_and_retrieved(
    tmp_path: Path,
) -> None:
    preprocessing = tmp_path / "preprocessing"
    models = tmp_path / "models"

    preprocessing.mkdir()
    models.mkdir()

    (preprocessing / "feature_scaling.md").write_text(
        """
# Feature Scaling

Standard scaling is commonly appropriate for logistic regression.
Tree-based models generally do not require feature scaling.
""".strip(),
        encoding="utf-8",
    )

    (models / "logistic_regression.md").write_text(
        """
# Logistic Regression

Logistic regression is a supervised classification algorithm.
It is commonly used as an interpretable baseline.
""".strip(),
        encoding="utf-8",
    )

    loader = MarkdownKnowledgeLoader(tmp_path)
    items = loader.load()

    retriever = InMemoryKnowledgeRetriever(items)

    results = retriever.retrieve(
        "classification logistic regression scaling"
    )

    assert len(results) == 2

    filenames = [
        item.metadata["filename"]
        for item in results
    ]

    assert "feature_scaling.md" in filenames
    assert "logistic_regression.md" in filenames