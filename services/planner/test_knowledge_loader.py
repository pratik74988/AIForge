from pathlib import Path

import pytest

from .knowledge_loader import MarkdownKnowledgeLoader


def test_loads_markdown_documents(tmp_path: Path) -> None:
    (tmp_path / "classification.md").write_text(
        "# Classification\n\nUse stratified splitting for imbalanced classes.",
        encoding="utf-8",
    )
    (tmp_path / "preprocessing.md").write_text(
        "# Preprocessing\n\nImpute missing numerical values.",
        encoding="utf-8",
    )

    loader = MarkdownKnowledgeLoader(tmp_path)

    items = loader.load()

    assert len(items) == 2
    assert items[0].content.startswith("# Classification")
    assert items[1].content.startswith("# Preprocessing")


def test_loads_documents_recursively(tmp_path: Path) -> None:
    preprocessing = tmp_path / "preprocessing"
    models = tmp_path / "models"

    preprocessing.mkdir()
    models.mkdir()

    (preprocessing / "missing_values.md").write_text(
        "Missing value guidance",
        encoding="utf-8",
    )

    (models / "logistic_regression.md").write_text(
        "Logistic regression guidance",
        encoding="utf-8",
    )

    loader = MarkdownKnowledgeLoader(tmp_path)

    items = loader.load()

    assert len(items) == 2

    filenames = {
        item.metadata["filename"]
        for item in items
    }

    assert filenames == {
        "missing_values.md",
        "logistic_regression.md",
    }


def test_loads_documents_in_deterministic_order(tmp_path: Path) -> None:
    preprocessing = tmp_path / "preprocessing"
    models = tmp_path / "models"

    preprocessing.mkdir()
    models.mkdir()

    (models / "z_model.md").write_text(
        "Z model",
        encoding="utf-8",
    )

    (preprocessing / "a_preprocessing.md").write_text(
        "A preprocessing",
        encoding="utf-8",
    )

    loader = MarkdownKnowledgeLoader(tmp_path)

    items = loader.load()

    assert [
        item.metadata["relative_path"]
        for item in items
    ] == [
        "models/z_model.md",
        "preprocessing/a_preprocessing.md",
    ]


def test_ignores_non_markdown_files(tmp_path: Path) -> None:
    preprocessing = tmp_path / "preprocessing"
    preprocessing.mkdir()

    (preprocessing / "missing_values.md").write_text(
        "Missing value guidance",
        encoding="utf-8",
    )

    (preprocessing / "notes.txt").write_text(
        "This should not be loaded",
        encoding="utf-8",
    )

    (preprocessing / "config.yaml").write_text(
        "something: true",
        encoding="utf-8",
    )

    loader = MarkdownKnowledgeLoader(tmp_path)

    items = loader.load()

    assert len(items) == 1
    assert items[0].metadata["filename"] == "missing_values.md"


def test_ignores_empty_markdown_files(tmp_path: Path) -> None:
    (tmp_path / "empty.md").write_text(
        "   ",
        encoding="utf-8",
    )

    (tmp_path / "valid.md").write_text(
        "Valid knowledge",
        encoding="utf-8",
    )

    loader = MarkdownKnowledgeLoader(tmp_path)

    items = loader.load()

    assert len(items) == 1
    assert items[0].content == "Valid knowledge"


def test_preserves_metadata(tmp_path: Path) -> None:
    preprocessing = tmp_path / "preprocessing"
    preprocessing.mkdir()

    document = preprocessing / "classification.md"

    document.write_text(
        "Classification guidance",
        encoding="utf-8",
    )

    loader = MarkdownKnowledgeLoader(tmp_path)

    items = loader.load()

    assert items[0].metadata is not None
    assert items[0].metadata["filename"] == "classification.md"
    assert items[0].metadata["source"] == str(document)
    assert items[0].metadata["relative_path"] == (
        "preprocessing/classification.md"
    )


def test_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    loader = MarkdownKnowledgeLoader(tmp_path)

    items = loader.load()

    assert items == []


def test_missing_directory_raises_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist"

    loader = MarkdownKnowledgeLoader(missing_path)

    with pytest.raises(FileNotFoundError):
        loader.load()


def test_file_path_raises_error(tmp_path: Path) -> None:
    file_path = tmp_path / "knowledge.md"

    file_path.write_text(
        "Knowledge",
        encoding="utf-8",
    )

    loader = MarkdownKnowledgeLoader(file_path)

    with pytest.raises(NotADirectoryError):
        loader.load()