import pytest
from src.ingestion.document_loader import Document, DocumentLoader
from src.ingestion.text_splitter import RecursiveCharacterSplitter


def test_document_loader(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello World! This is a test document.")

    doc = DocumentLoader.load_file(test_file)
    assert doc.id == "test.txt"
    assert "Hello World" in doc.content
    assert doc.metadata["filename"] == "test.txt"


def test_recursive_character_splitter():
    text = "Paragraph 1 is here with some words.\n\nParagraph 2 is here with more interesting text.\n\nParagraph 3 is also here."
    doc = Document(id="doc1", content=text, metadata={"filename": "doc1.md"})

    splitter = RecursiveCharacterSplitter(chunk_size=40, chunk_overlap=10)
    chunks = splitter.split_documents([doc])

    assert len(chunks) >= 2
    assert all(c.metadata["parent_id"] == "doc1" for c in chunks)
    assert all(isinstance(c.content, str) for c in chunks)
