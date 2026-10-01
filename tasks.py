"""
The actual indexing work, run inside RQ worker processes (see worker.py)
instead of the original app's bare ThreadPoolExecutor. That matters because:
  - jobs now survive an app restart (they're durable in Redis, not just an
    in-memory thread pool that's wiped on crash/redeploy)
  - indexing load is isolated from the web process, so a slow LLM graph
    extraction pass can't starve Flask's request handling
  - failed jobs are visible/retryable via RQ instead of silently vanishing
    into a logged exception
"""
import logging
import os

from config import Config
import store
import bm25_store
import ingestion
import graph_engine
import tenant_registry


def index_document_task(filepath: str, size: int, modified_time: str, tenant_id: str = None):
    tenant_id = tenant_id or Config.DEFAULT_TENANT
    filename = os.path.basename(filepath)
    store.set_indexing_status(tenant_id, filepath, "processing", 0)

    try:
        components = tenant_registry.get_tenant(tenant_id)
        parent_retriever = components["parent_retriever"]
        vectorstore = components["vectorstore"]

        logging.info(f"[{tenant_id}] Loading '{filename}'...")
        raw_docs = ingestion.load_source(filepath)
        if not raw_docs:
            store.set_indexing_status(tenant_id, filepath, "failed", 0, "No text extracted")
            return {"status": "failed", "reason": "no_text_extracted"}

        for i, doc in enumerate(raw_docs):
            doc.metadata.update(
                {"source": filepath, "size": size, "modifiedTime": str(modified_time), "page": i + 1}
            )

        # --- replace any previous version of this file everywhere ---
        _delete_existing(tenant_id, filepath, components)

        # --- vector store (parent/child) ---
        parent_retriever.add_documents(raw_docs, ids=None)
        store.set_indexing_status(tenant_id, filepath, "processing", 35)

        # --- BM25 keyword index: index the same child-sized chunks so both
        #     retrieval paths operate at comparable granularity ---
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHILD_CHUNK_SIZE, chunk_overlap=Config.CHILD_CHUNK_OVERLAP
        )
        child_chunks = child_splitter.split_documents(raw_docs)
        bm25_store.add_documents(tenant_id, filepath, child_chunks)
        store.set_indexing_status(tenant_id, filepath, "processing", 55)

        # --- knowledge graph ---
        nodes, rels = 0, 0
        if Config.ENABLE_GRAPH:
            nodes, rels = _index_graph(tenant_id, filepath, raw_docs, components)

        store.set_indexing_status(tenant_id, filepath, "completed", 100)
        logging.info(
            f"[{tenant_id}] Finished '{filename}': {len(child_chunks)} chunks, "
            f"{nodes} graph nodes, {rels} graph relationships."
        )
        return {"status": "completed", "chunks": len(child_chunks), "graph_nodes": nodes, "graph_rels": rels}

    except Exception as e:
        logging.error(f"[{tenant_id}] Failed indexing '{filepath}': {e}", exc_info=True)
        store.set_indexing_status(tenant_id, filepath, "failed", 0, str(e))
        raise  # let RQ record/retry the failure rather than swallowing it


def _delete_existing(tenant_id: str, filepath: str, components):
    vectorstore = components["vectorstore"]
    parent_retriever = components["parent_retriever"]
    try:
        existing = vectorstore._collection.get(where={"source": filepath}, include=["metadatas"])
        existing_ids = existing["ids"]
        if existing_ids:
            parent_ids = {m["doc_id"] for m in existing["metadatas"] if "doc_id" in m}
            vectorstore.delete(ids=existing_ids)
            if parent_ids:
                parent_retriever.docstore.mdelete(list(parent_ids))
            logging.info(f"[{tenant_id}] Deleted {len(existing_ids)} old vector chunks for '{filepath}'.")
    except Exception as e:
        logging.warning(f"[{tenant_id}] Could not clean up old vector chunks for '{filepath}': {e}")

    bm25_store.delete_source(tenant_id, filepath)

    if Config.ENABLE_GRAPH:
        try:
            graph_engine.delete_source(filepath, tenant_id)
        except Exception as e:
            logging.warning(f"[{tenant_id}] Could not clean up old graph nodes for '{filepath}': {e}")


def _index_graph(tenant_id: str, filepath: str, raw_docs, components):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_experimental.graph_transformers import LLMGraphTransformer

    graph_llm = components["graph_llm"]
    graph_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.GRAPH_CHUNK_SIZE, chunk_overlap=Config.GRAPH_CHUNK_OVERLAP
    )
    graph_chunks = graph_splitter.split_documents(raw_docs)

    total_nodes = total_rels = 0
    transformer = LLMGraphTransformer(llm=graph_llm)
    batch_size = Config.GRAPH_BATCH_SIZE
    total_batches = (len(graph_chunks) + batch_size - 1) // batch_size

    for i in range(0, len(graph_chunks), batch_size):
        batch = graph_chunks[i : i + batch_size]
        batch_num = i // batch_size + 1
        try:
            nodes, rels = graph_engine.index_chunks(transformer, batch, filepath, tenant_id)
            total_nodes += nodes
            total_rels += rels
            progress = 55 + int((batch_num / max(total_batches, 1)) * 45)
            store.set_indexing_status(tenant_id, filepath, "processing", progress)
        except Exception as e:
            logging.error(f"[{tenant_id}] Graph batch {batch_num}/{total_batches} failed: {e}", exc_info=True)
            continue

    return total_nodes, total_rels
