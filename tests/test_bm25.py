import shutil
import tempfile

import pytest
from langchain_core.documents import Document as LC_Doc


@pytest.fixture
def bm25(monkeypatch):
    tmp_dir = tempfile.mkdtemp()
    from config import Config
    monkeypatch.setattr(Config, "BM25_INDEX_DIR", tmp_dir)
    import importlib
    import bm25_store as bm25_module
    importlib.reload(bm25_module)
    yield bm25_module
    shutil.rmtree(tmp_dir, ignore_errors=True)


# NOTE: BM25's classic IDF formula, log((N - n + 0.5) / (n + 0.5)), degenerates
# to exactly zero when a term appears in exactly half of a 2-document corpus
# (n=1, N=2) - a quirk of tiny corpora, not something real usage (hundreds of
# chunks) hits. These tests use >=3 documents so IDF behaves meaningfully.

def test_search_finds_relevant_chunk(bm25):
    docs = [
        LC_Doc(page_content="The quarterly budget for Flower Ltd was 2.4 million dollars.", metadata={"source": "a.pdf"}),
        LC_Doc(page_content="Shlok is pursuing a degree in computer science.", metadata={"source": "c.pdf"}),
        LC_Doc(page_content="Unrelated text about gardening and roses.", metadata={"source": "b.pdf"}),
    ]
    bm25.add_documents("tenantA", "a.pdf", [docs[0]])
    bm25.add_documents("tenantA", "c.pdf", [docs[1]])
    bm25.add_documents("tenantA", "b.pdf", [docs[2]])

    results = bm25.search("tenantA", "Flower Ltd budget", top_k=5)
    assert len(results) >= 1
    assert "Flower Ltd" in results[0][0]["content"]


def test_delete_source_removes_its_chunks(bm25):
    bm25.add_documents("tenantA", "b.pdf", [LC_Doc(page_content="gardening and roses content", metadata={"source": "b.pdf"})])
    bm25.add_documents("tenantA", "c.pdf", [LC_Doc(page_content="an unrelated finance report", metadata={"source": "c.pdf"})])
    bm25.add_documents("tenantA", "d.pdf", [LC_Doc(page_content="a third unrelated document", metadata={"source": "d.pdf"})])
    assert bm25.search("tenantA", "gardening roses", top_k=5)

    bm25.delete_source("tenantA", "b.pdf")
    assert bm25.search("tenantA", "gardening roses", top_k=5) == []


def test_tenant_isolation(bm25):
    bm25.add_documents("tenantA", "a.pdf", [LC_Doc(page_content="alpha secret content", metadata={"source": "a.pdf"})])
    bm25.add_documents("tenantB", "b.pdf", [LC_Doc(page_content="beta secret content", metadata={"source": "b.pdf"})])

    assert bm25.search("tenantA", "beta secret", top_k=5) == []
    assert bm25.search("tenantB", "alpha secret", top_k=5) == []
