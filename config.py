"""
Central configuration. Everything is read from the environment (via .env in
development) so the app is no longer hardcoded to one machine's Ollama/Neo4j
setup. Copy .env.example to .env and edit it.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class Config:
    # ---- Storage locations ----
    KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base_storage")
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "rag_db")
    CHROMA_COLLECTION_PREFIX = os.getenv("CHROMA_COLLECTION_PREFIX", "documents")
    BM25_INDEX_DIR = os.getenv("BM25_INDEX_DIR", "bm25_indexes")
    APP_DB_PATH = os.getenv("APP_DB_PATH", "app_data.sqlite3")

    # ---- Multi-tenancy ----
    DEFAULT_TENANT = os.getenv("DEFAULT_TENANT", "default")

    # ---- LLM provider (see llm_providers.py) ----
    # "ollama" | "openai" | "openai_compatible" | "anthropic"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
    LLM_MODEL = os.getenv("LLM_MODEL", "DeepSeek-R1:latest")
    LLM_API_BASE = os.getenv("LLM_API_BASE")  # for ollama / openai_compatible
    LLM_API_KEY = os.getenv("LLM_API_KEY")    # for openai / anthropic / openai_compatible

    # Optionally use a smaller/cheaper model just for graph extraction & Cypher gen
    GRAPH_LLM_PROVIDER = os.getenv("GRAPH_LLM_PROVIDER", LLM_PROVIDER)
    GRAPH_LLM_MODEL = os.getenv("GRAPH_LLM_MODEL", LLM_MODEL)

    # ---- Embeddings & reranking ----
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
    RE_RANKER_MODEL_NAME = os.getenv("RE_RANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")  # auto | cpu | cuda

    # ---- Retrieval / chunking ----
    SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".docx", ".csv", ".xlsx", ".xls")
    PARENT_CHUNK_SIZE = _int("PARENT_CHUNK_SIZE", 4000)
    PARENT_CHUNK_OVERLAP = _int("PARENT_CHUNK_OVERLAP", 400)
    CHILD_CHUNK_SIZE = _int("CHILD_CHUNK_SIZE", 800)
    CHILD_CHUNK_OVERLAP = _int("CHILD_CHUNK_OVERLAP", 150)

    RETRIEVAL_TOP_N = _int("RETRIEVAL_TOP_N", 30)
    RE_RANKED_TOP_N = _int("RE_RANKED_TOP_N", 8)

    # Ensemble fusion weights (vector vs keyword/BM25)
    ENSEMBLE_VECTOR_WEIGHT = _float("ENSEMBLE_VECTOR_WEIGHT", 0.6)
    ENSEMBLE_BM25_WEIGHT = _float("ENSEMBLE_BM25_WEIGHT", 0.4)

    # Below this cross-encoder relevance score, a chunk is treated as noise
    MIN_RELEVANCE_SCORE = _float("MIN_RELEVANCE_SCORE", 0.15)
    # If NO retrieved chunk clears the bar, refuse to answer from documents
    ENABLE_GROUNDING_GATE = _bool("ENABLE_GROUNDING_GATE", True)

    # Router: if ambiguous/low-confidence, query both engines instead of guessing
    ROUTER_FALLBACK_TO_HYBRID = _bool("ROUTER_FALLBACK_TO_HYBRID", True)

    # Rewrite follow-up questions into standalone queries before retrieval
    ENABLE_QUERY_REWRITING = _bool("ENABLE_QUERY_REWRITING", True)

    # ---- OCR ----
    ENABLE_OCR_FALLBACK = _bool("ENABLE_OCR_FALLBACK", True)
    OCR_MIN_CHARS_PER_PAGE = _int("OCR_MIN_CHARS_PER_PAGE", 20)

    # ---- Graph (Neo4j) ----
    ENABLE_GRAPH = _bool("ENABLE_GRAPH", True)
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
    GRAPH_BATCH_SIZE = _int("GRAPH_BATCH_SIZE", 3)
    GRAPH_CHUNK_SIZE = _int("GRAPH_CHUNK_SIZE", 2000)
    GRAPH_CHUNK_OVERLAP = _int("GRAPH_CHUNK_OVERLAP", 200)

    # ---- Task queue (Redis + RQ) ----
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    RQ_QUEUE_NAME = os.getenv("RQ_QUEUE_NAME", "indexing")
    # Parent-doc store: redis (persists across restarts) or memory (dev only)
    DOCSTORE_BACKEND = os.getenv("DOCSTORE_BACKEND", "redis")

    # ---- Server & chat ----
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = _int("PORT", 5000)
    MAX_HISTORY_TURNS = _int("MAX_HISTORY_TURNS", 10)

    @classmethod
    def validate(cls):
        problems = []
        if cls.LLM_PROVIDER in ("openai", "anthropic") and not cls.LLM_API_KEY:
            problems.append(f"LLM_PROVIDER={cls.LLM_PROVIDER} requires LLM_API_KEY")
        if cls.ENABLE_GRAPH and not cls.NEO4J_PASSWORD:
            problems.append("ENABLE_GRAPH=true but NEO4J_PASSWORD is unset")
        return problems
