"""
A minimal, dependency-light BM25 keyword index, kept alongside the vector
store so exact names/IDs/numbers that embeddings tend to smear over are
still retrievable. Dense embeddings and BM25 fail on different kinds of
queries, so combining them (see retrieval.EnsembleRetriever usage) covers
more ground than either alone.

Persisted as one JSON corpus file per tenant so it survives restarts;
rebuilt into an in-memory rank_bm25 index lazily and cached per tenant.
"""
import json
import os
import re
import threading

from rank_bm25 import BM25Okapi

from config import Config

_token_re = re.compile(r"[A-Za-z0-9_]+")
_cache = {}
_cache_lock = threading.Lock()


def _tokenize(text: str):
    return _token_re.findall(text.lower())


def _corpus_path(tenant_id: str) -> str:
    os.makedirs(Config.BM25_INDEX_DIR, exist_ok=True)
    return os.path.join(Config.BM25_INDEX_DIR, f"{tenant_id}.jsonl")


def _load_corpus(tenant_id: str):
    path = _corpus_path(tenant_id)
    if not os.path.exists(path):
        return []
    docs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def _save_corpus(tenant_id: str, docs):
    path = _corpus_path(tenant_id)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    os.replace(tmp_path, path)


def add_documents(tenant_id: str, source: str, chunks):
    """chunks: list of langchain Document. Replaces any existing entries for `source`."""
    docs = [d for d in _load_corpus(tenant_id) if d.get("metadata", {}).get("source") != source]
    for c in chunks:
        docs.append({"content": c.page_content, "metadata": dict(c.metadata)})
    _save_corpus(tenant_id, docs)
    with _cache_lock:
        _cache.pop(tenant_id, None)  # invalidate cached index


def delete_source(tenant_id: str, source: str):
    docs = [d for d in _load_corpus(tenant_id) if d.get("metadata", {}).get("source") != source]
    _save_corpus(tenant_id, docs)
    with _cache_lock:
        _cache.pop(tenant_id, None)


def _get_index(tenant_id: str):
    with _cache_lock:
        cached = _cache.get(tenant_id)
        docs = _load_corpus(tenant_id)
        if cached and cached["size"] == len(docs):
            return cached["bm25"], docs
        if not docs:
            _cache.pop(tenant_id, None)
            return None, []
        tokenized = [_tokenize(d["content"]) for d in docs]
        bm25 = BM25Okapi(tokenized)
        _cache[tenant_id] = {"bm25": bm25, "size": len(docs)}
        return bm25, docs


def search(tenant_id: str, query: str, top_k: int = 10):
    """Returns list of (langchain-style dict, score) sorted best-first."""
    bm25, docs = _get_index(tenant_id)
    if bm25 is None:
        return []
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [(d, s) for d, s in ranked[:top_k] if s > 0]
