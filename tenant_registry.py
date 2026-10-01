"""
Everything in the original app was a single global vectorstore/docstore/
retriever built once at import time - fine for one knowledge base, unusable
for serving several. This module builds those components per tenant_id on
first use and caches them, so `POST /ask` (tenant=acme) and
`POST /ask` (tenant=globex) hit completely separate Chroma collections,
BM25 corpora, and (via a tenant_id property filter) Neo4j subgraphs, while
sharing one running process, one embedding model in memory, and one Neo4j
connection.
"""
import logging
import threading

from config import Config

_lock = threading.Lock()
_registry = {}
_embeddings = None
_cross_encoder = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        device = Config.EMBEDDING_DEVICE
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        _embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL_NAME, model_kwargs={"device": device}
        )
    return _embeddings


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        _cross_encoder = HuggingFaceCrossEncoder(model_name=Config.RE_RANKER_MODEL_NAME)
    return _cross_encoder


def _build_docstore(tenant_id: str):
    if Config.DOCSTORE_BACKEND == "redis":
        from langchain_community.storage import RedisStore
        import redis
        client = redis.Redis.from_url(Config.REDIS_URL)
        # namespace keys per tenant so parent docs never collide across tenants
        return RedisStore(client=client, namespace=f"parentdocs:{tenant_id}")
    from langchain.storage import InMemoryStore
    logging.warning(
        f"DOCSTORE_BACKEND=memory for tenant '{tenant_id}': parent documents will NOT "
        f"survive a restart. Use DOCSTORE_BACKEND=redis for production."
    )
    return InMemoryStore()


def get_tenant(tenant_id: str = None):
    tenant_id = tenant_id or Config.DEFAULT_TENANT
    with _lock:
        if tenant_id in _registry:
            return _registry[tenant_id]

        from langchain_chroma import Chroma
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.retrievers import ParentDocumentRetriever, MultiQueryRetriever

        import llm_providers

        vectorstore = Chroma(
            collection_name=f"{Config.CHROMA_COLLECTION_PREFIX}_{tenant_id}",
            persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
            embedding_function=_get_embeddings(),
        )
        docstore = _build_docstore(tenant_id)
        parent_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=RecursiveCharacterTextSplitter(
                chunk_size=Config.CHILD_CHUNK_SIZE, chunk_overlap=Config.CHILD_CHUNK_OVERLAP
            ),
            parent_splitter=RecursiveCharacterTextSplitter(
                chunk_size=Config.PARENT_CHUNK_SIZE, chunk_overlap=Config.PARENT_CHUNK_OVERLAP
            ),
        )

        main_llm = llm_providers.get_main_llm()
        graph_llm = llm_providers.get_graph_llm() if Config.ENABLE_GRAPH else None
        multi_query_retriever = MultiQueryRetriever.from_llm(retriever=parent_retriever, llm=main_llm)

        components = {
            "tenant_id": tenant_id,
            "vectorstore": vectorstore,
            "docstore": docstore,
            "parent_retriever": parent_retriever,
            "multi_query_retriever": multi_query_retriever,
            "cross_encoder": _get_cross_encoder(),
            "main_llm": main_llm,
            "graph_llm": graph_llm,
        }
        _registry[tenant_id] = components
        logging.info(f"Initialized retrieval components for tenant '{tenant_id}'.")
        return components


def list_tenants():
    with _lock:
        return list(_registry.keys())
