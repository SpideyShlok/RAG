"""
The accuracy-critical pieces of the pipeline, kept as small pure(ish)
functions so they can be unit-tested without a live LLM/vector DB/Neo4j:

  1. reciprocal_rank_fusion  - merges dense (vector) and sparse (BM25)
     rankings into one list. Dense embeddings miss exact names/IDs/numbers;
     BM25 misses paraphrases and synonyms. Fusing both beats either alone.
  2. grounding_gate          - after cross-encoder reranking, refuses to let
     the LLM answer from chunks that scored below a relevance floor, instead
     of silently feeding it noise it will confidently hallucinate around.
  3. parse_router_output     - classifies "vector" vs "graph" vs "ambiguous"
     instead of defaulting blindly, so callers can fall back to hybrid mode
     on low-confidence routes.
  4. rewrite_query_with_history - turns a follow-up ("what about them?")
     into a standalone query before it ever reaches retrieval, since the
     original file only fed history to the *generation* prompt, never to
     retrieval, so follow-ups retrieved poorly.
"""
import hashlib
import logging
from collections import defaultdict

from langchain_core.documents import Document as LC_Doc

from config import Config


# ------------------------------------------------------------------ fusion --

def _doc_key(content: str, metadata: dict) -> str:
    source = metadata.get("source", "")
    digest = hashlib.sha1((content or "").strip().encode("utf-8", "ignore")).hexdigest()[:16]
    return f"{source}:{digest}"


def reciprocal_rank_fusion(ranked_lists, weights=None, k: int = 60):
    """
    ranked_lists: list of lists of langchain Document, each already ordered
    best-first by its own retriever.
    weights: optional per-list weight, same length as ranked_lists.
    Returns a single list of Document, deduplicated, best-first.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores = defaultdict(float)
    doc_by_key = {}
    for docs, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(docs):
            key = _doc_key(doc.page_content, doc.metadata)
            scores[key] += weight * (1.0 / (k + rank + 1))
            doc_by_key.setdefault(key, doc)

    ordered_keys = sorted(scores.keys(), key=lambda kk: scores[kk], reverse=True)
    return [doc_by_key[kk] for kk in ordered_keys]


def bm25_hits_to_documents(hits):
    """bm25_store.search() returns (dict, score) pairs; convert to Documents."""
    out = []
    for d, _score in hits:
        out.append(LC_Doc(page_content=d["content"], metadata=d.get("metadata", {})))
    return out


# -------------------------------------------------------------- grounding --

def score_with_cross_encoder(cross_encoder_model, question: str, docs):
    """cross_encoder_model is a sentence-transformers CrossEncoder (or shares
    its .predict([(q, text), ...]) interface, as HuggingFaceCrossEncoder does)."""
    if not docs:
        return []
    pairs = [(question, d.page_content) for d in docs]
    raw_scores = cross_encoder_model.score(pairs) if hasattr(cross_encoder_model, "score") \
        else cross_encoder_model.predict(pairs)
    return list(zip(docs, [float(s) for s in raw_scores]))


def grounding_gate(scored_docs, min_score: float = None, top_n: int = None):
    """
    scored_docs: list of (Document, score), any order.
    Returns (grounded: bool, kept_docs: list[Document]) best-first.
    grounded=False means nothing cleared the relevance floor -> caller should
    refuse to answer from documents rather than pass weak/irrelevant chunks
    to the LLM.
    """
    min_score = Config.MIN_RELEVANCE_SCORE if min_score is None else min_score
    top_n = Config.RE_RANKED_TOP_N if top_n is None else top_n

    ranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    kept = [doc for doc, score in ranked if score >= min_score][:top_n]

    if not kept and Config.ENABLE_GROUNDING_GATE:
        return False, []
    if not kept:  # gate disabled: fall back to best-effort top_n regardless of score
        kept = [doc for doc, _ in ranked[:top_n]]
    return True, kept


# ------------------------------------------------------------------ router --

def parse_router_output(raw_output: str) -> str:
    """Returns 'graph', 'vector', or 'ambiguous' (never guesses silently)."""
    text = (raw_output or "").lower()
    has_graph = "graph" in text
    has_vector = "vector" in text
    if has_graph and not has_vector:
        return "graph"
    if has_vector and not has_graph:
        return "vector"
    logging.warning(f"Ambiguous router output: {raw_output!r}")
    return "ambiguous"


def resolve_route(raw_output: str) -> tuple:
    """
    Returns (route, use_hybrid_fallback).
    On ambiguous output, either fall back to hybrid (query both engines) or
    default to vector, depending on Config.ROUTER_FALLBACK_TO_HYBRID.
    """
    route = parse_router_output(raw_output)
    if route != "ambiguous":
        return route, False
    if Config.ROUTER_FALLBACK_TO_HYBRID:
        return "hybrid", True
    return "vector", False


# ------------------------------------------------------- query rewriting --

_CONDENSE_TEMPLATE = """Given the conversation history and a follow-up question, rewrite the \
follow-up into a standalone question that contains all the context needed to search for an \
answer on its own. If the follow-up is already standalone, return it unchanged. Output ONLY \
the rewritten question, nothing else.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""


def rewrite_query_with_history(llm, question: str, history_messages) -> str:
    """
    history_messages: list of langchain HumanMessage/AIMessage (may be empty).
    Falls back to the original question on any error or when rewriting is
    disabled/there's no history to disambiguate against.
    """
    if not Config.ENABLE_QUERY_REWRITING or not history_messages:
        return question
    try:
        history_text = "\n".join(
            f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}"
            for msg in history_messages[-6:]
        )
        prompt = _CONDENSE_TEMPLATE.format(history=history_text, question=question)
        result = llm.invoke(prompt)
        rewritten = getattr(result, "content", result)
        rewritten = (rewritten or "").strip().strip('"')
        return rewritten if rewritten else question
    except Exception as e:
        logging.warning(f"Query rewriting failed, using original question: {e}")
        return question
