import json
import logging
import os

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import Config
import store
import bm25_store
import graph_engine
import retrieval
import tenant_registry
from queue_utils import get_queue

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
os.makedirs(Config.KNOWLEDGE_BASE_DIR, exist_ok=True)
store.init_db()

for problem in Config.validate():
    logging.warning(f"Config warning: {problem}")

app = Flask(__name__, static_folder="templates", static_url_path="")
CORS(app)

# ---------------------------------------------------------------- prompts --

ROUTER_PROMPT = PromptTemplate.from_template(
    """You are a routing expert. Decide which system should answer this question.

GRAPH DATABASE (Neo4j): specific facts about entities, properties, relationships.
  e.g. "What course is Shlok pursuing?", "Who works for Flower Ltd?"

VECTOR DATABASE (documents): summaries, explanations, open-ended or thematic questions.
  e.g. "Summarize the audit report", "What are the main risks discussed?"

Question: {question}

Output ONLY ONE WORD: "graph" or "vector"."""
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a knowledgeable assistant. Answer using ONLY the context below "
            "(which may include document excerpts and/or knowledge-graph facts).\n"
            "Rules:\n"
            "1. Base your answer entirely on the context.\n"
            "2. Be specific and cite relevant details.\n"
            "3. If the context is contradictory, mention both viewpoints.\n"
            "4. Use conversation history for continuity, but answer from the context.\n\n"
            "Context:\n{context_str}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def _format_docs(docs) -> str:
    if not docs:
        return "No relevant documents found."
    return "\n\n---\n\n".join(f"Document {i + 1}:\n{d.page_content}" for i, d in enumerate(docs))


def _to_lc_messages(turns):
    out = []
    for t in turns:
        if t["role"] == "user":
            out.append(HumanMessage(content=t["content"]))
        elif t["role"] == "assistant":
            out.append(AIMessage(content=t["content"]))
    return out


def _resolve_tenant() -> str:
    header_val = request.headers.get("X-Tenant-ID")
    if header_val:
        return header_val
    if request.is_json:
        body_val = (request.get_json(silent=True) or {}).get("tenant")
        if body_val:
            return body_val
    args_val = request.args.get("tenant")
    if args_val:
        return args_val
    return Config.DEFAULT_TENANT


def _llm_text(result) -> str:
    return getattr(result, "content", result) or ""


# --------------------------------------------------------------- ingestion --

def _enqueue_index(filepath: str, tenant_id: str):
    stat = os.stat(filepath)
    store.set_indexing_status(tenant_id, filepath, "queued", 0)
    job = get_queue().enqueue(
        "tasks.index_document_task",
        filepath,
        stat.st_size,
        str(stat.st_mtime),
        tenant_id,
        job_timeout=Config.__dict__.get("JOB_TIMEOUT", 1800),
    )
    return job.id


@app.route("/")
def home():
    index_path = os.path.join(app.static_folder or "templates", "index.html")
    if os.path.exists(index_path):
        return send_from_directory(app.static_folder, "index.html")
    return jsonify({"status": "ok", "message": "RAG API is running. See README for endpoints."})


@app.route("/upload", methods=["POST"])
def upload_files_endpoint():
    tenant_id = _resolve_tenant()
    if "files" not in request.files:
        return jsonify({"status": "error", "message": "No files part"}), 400
    files = request.files.getlist("files")
    job_ids, saved = [], []
    for file in files:
        if not file or file.filename == "":
            continue
        try:
            filepath = os.path.join(Config.KNOWLEDGE_BASE_DIR, tenant_id, file.filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            file.save(filepath)
            job_ids.append(_enqueue_index(filepath, tenant_id))
            saved.append(filepath)
        except Exception as e:
            logging.error(f"Error saving file {file.filename}: {e}")
    return jsonify({"status": "success", "processed": len(saved), "files": saved, "job_ids": job_ids}), 200


@app.route("/ingest-url", methods=["POST"])
def ingest_url_endpoint():
    """New: index a web page directly instead of requiring a local file."""
    tenant_id = _resolve_tenant()
    url = (request.json or {}).get("url", "").strip()
    if not url.lower().startswith(("http://", "https://")):
        return jsonify({"status": "error", "message": "A valid http(s) url is required"}), 400
    store.set_indexing_status(tenant_id, url, "queued", 0)
    job = get_queue().enqueue("tasks.index_document_task", url, 0, "0", tenant_id)
    return jsonify({"status": "success", "url": url, "job_id": job.id}), 202


@app.route("/set-folder", methods=["POST"])
def set_folder_endpoint():
    tenant_id = _resolve_tenant()
    folder_path = (request.json or {}).get("folder_path")
    if not folder_path or not os.path.isdir(folder_path):
        return jsonify({"status": "error", "message": "Invalid folder_path"}), 400
    count = 0
    try:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(Config.SUPPORTED_EXTENSIONS):
                filepath = os.path.join(folder_path, filename)
                if os.path.isfile(filepath):
                    _enqueue_index(filepath, tenant_id)
                    count += 1
        return jsonify({"status": "success", "message": f"Queued {count} files for indexing."}), 202
    except Exception as e:
        logging.error(f"Error reading folder {folder_path}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/reindex", methods=["POST"])
def reindex_endpoint():
    tenant_id = _resolve_tenant()
    filepath = (request.json or {}).get("filepath")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"status": "error", "message": "Invalid filepath"}), 400
    job_id = _enqueue_index(filepath, tenant_id)
    return jsonify({"status": "success", "job_id": job_id}), 202


@app.route("/delete-file", methods=["DELETE"])
def delete_file_endpoint():
    tenant_id = _resolve_tenant()
    filepath = (request.json or {}).get("filepath")
    if not filepath:
        return jsonify({"status": "error", "message": "No filepath provided"}), 400

    deleted_from = []
    components = tenant_registry.get_tenant(tenant_id)
    try:
        existing = components["vectorstore"]._collection.get(where={"source": filepath}, include=["metadatas"])
        if existing["ids"]:
            components["vectorstore"].delete(ids=existing["ids"])
            deleted_from.append("vector_store")
    except Exception as e:
        logging.error(f"Error deleting from vector store: {e}")

    try:
        bm25_store.delete_source(tenant_id, filepath)
        deleted_from.append("bm25_index")
    except Exception as e:
        logging.error(f"Error deleting from BM25 index: {e}")

    if Config.ENABLE_GRAPH:
        try:
            graph_engine.delete_source(filepath, tenant_id)
            deleted_from.append("graph_store")
        except Exception as e:
            logging.error(f"Error deleting from graph store: {e}")

    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            deleted_from.append("filesystem")
        except Exception as e:
            logging.error(f"Error deleting file: {e}")

    store.delete_indexing_status(tenant_id, filepath)
    return jsonify({"status": "success", "deleted_from": deleted_from, "filepath": filepath}), 200


@app.route("/check-files", methods=["GET"])
def check_files_endpoint():
    tenant_id = _resolve_tenant()
    try:
        components = tenant_registry.get_tenant(tenant_id)
        all_docs = components["vectorstore"]._collection.get(include=["metadatas"])
        unique_files = sorted({m["source"] for m in all_docs["metadatas"] if m and "source" in m})
        return jsonify({"files": unique_files})
    except Exception as e:
        logging.error(f"Failed to retrieve file list: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/indexing-status", methods=["GET"])
def indexing_status_endpoint():
    tenant_id = _resolve_tenant()
    filepath = request.args.get("filepath")
    return jsonify(store.get_indexing_status(tenant_id, filepath))


# -------------------------------------------------------------------- ask --

@app.route("/ask", methods=["POST"])
def ask_endpoint():
    data = request.json or {}
    question = (data.get("prompt") or "").strip()
    if not question:
        return jsonify({"error": "Empty prompt"}), 400

    tenant_id = _resolve_tenant()
    session_id = request.headers.get("X-Session-ID", request.remote_addr)
    force_hybrid = bool(data.get("hybrid", False))

    def stream_response():
        components = tenant_registry.get_tenant(tenant_id)
        main_llm = components["main_llm"]
        graph_llm = components["graph_llm"]
        multi_query_retriever = components["multi_query_retriever"]
        cross_encoder = components["cross_encoder"]

        full_response = ""
        source_docs = []
        route_used = "hybrid" if force_hybrid else None

        try:
            turns = store.get_turns(tenant_id, session_id, limit_turns=Config.MAX_HISTORY_TURNS)
            history_messages = _to_lc_messages(turns)

            # 1. Rewrite follow-ups into standalone queries BEFORE retrieval.
            search_query = retrieval.rewrite_query_with_history(main_llm, question, history_messages)
            if search_query != question:
                logging.info(f"Rewrote query for retrieval: '{question}' -> '{search_query}'")

            # 2. Route, with a hybrid fallback on low-confidence classification.
            use_hybrid = force_hybrid
            route = "vector"
            if Config.ENABLE_GRAPH and not force_hybrid:
                raw = _llm_text(main_llm.invoke(ROUTER_PROMPT.format(question=search_query)))
                route, ambiguous = retrieval.resolve_route(raw)
                use_hybrid = ambiguous or route == "hybrid"
                if route == "hybrid":
                    route = "vector"  # vector path also fetches graph facts below
            route_used = "hybrid" if use_hybrid else route

            graph_answer = None
            if Config.ENABLE_GRAPH and (route == "graph" or use_hybrid):
                try:
                    graph_result = graph_engine.query_graph(graph_llm, search_query, tenant_id)
                    graph_answer = graph_result.get("result")
                except Exception as e:
                    logging.error(f"Graph query failed: {e}", exc_info=True)
                    graph_answer = None
                    if route == "graph":
                        route = "vector"  # graph engine unavailable -> fall back
                        use_hybrid = True

            if route == "graph" and not use_hybrid:
                # Pure graph answer, no document grounding gate applies.
                answer = graph_answer or "I couldn't find relevant information in the knowledge graph."
                full_response = answer
                yield f"data: {json.dumps({'token': answer})}\n\n"
            else:
                # 3. Ensemble retrieval: dense (vector) + sparse (BM25), fused.
                vector_docs = multi_query_retriever.invoke(search_query)
                bm25_hits = bm25_store.search(tenant_id, search_query, top_k=Config.RETRIEVAL_TOP_N)
                bm25_docs = retrieval.bm25_hits_to_documents(bm25_hits)
                fused = retrieval.reciprocal_rank_fusion(
                    [vector_docs, bm25_docs],
                    weights=[Config.ENSEMBLE_VECTOR_WEIGHT, Config.ENSEMBLE_BM25_WEIGHT],
                )

                # 4. Cross-encoder rerank + grounding gate.
                scored = retrieval.score_with_cross_encoder(cross_encoder, search_query, fused)
                grounded, kept_docs = retrieval.grounding_gate(scored)
                source_docs = kept_docs

                if not grounded and not graph_answer:
                    answer = (
                        "I don't have enough relevant, grounded information in the indexed "
                        "documents to answer that confidently."
                    )
                    full_response = answer
                    yield f"data: {json.dumps({'token': answer})}\n\n"
                else:
                    context_str = _format_docs(kept_docs)
                    if graph_answer:
                        context_str += f"\n\n--- Knowledge graph facts ---\n{graph_answer}"

                    chain = ANSWER_PROMPT | main_llm | StrOutputParser()
                    for chunk in chain.stream(
                        {"context_str": context_str, "chat_history": history_messages, "question": question}
                    ):
                        full_response += chunk
                        yield f"data: {json.dumps({'token': chunk})}\n\n"

            # --- sources ---
            if source_docs:
                header = "\n\n---\n**Sources:**\n"
                full_response += header
                yield f"data: {json.dumps({'token': header})}\n\n"
                seen = set()
                for doc in source_docs:
                    name = os.path.basename(doc.metadata.get("source", "Unknown Source"))
                    page = doc.metadata.get("page", "")
                    key = f"{name}_{page}"
                    if key in seen:
                        continue
                    seen.add(key)
                    entry = f"- {name}" + (f" (page {page})" if page else "") + "\n"
                    full_response += entry
                    yield f"data: {json.dumps({'token': entry})}\n\n"

            if route_used:
                debug = f"\n\n*[Retrieved from: {route_used}]*"
                full_response += debug
                yield f"data: {json.dumps({'token': debug})}\n\n"

            store.append_turn(tenant_id, session_id, "user", question)
            store.append_turn(tenant_id, session_id, "assistant", full_response)

        except Exception as e:
            logging.error(f"Error during /ask: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_response(), mimetype="text/event-stream")


@app.route("/search", methods=["POST"])
def search_endpoint():
    tenant_id = _resolve_tenant()
    data = request.json or {}
    query = (data.get("query") or "").strip()
    db_type = data.get("type", "vector")
    if not query:
        return jsonify({"error": "Empty query"}), 400

    components = tenant_registry.get_tenant(tenant_id)
    try:
        if db_type == "vector":
            docs = components["multi_query_retriever"].invoke(query)
            results = [
                {"content": d.page_content[:500], "metadata": d.metadata} for d in docs
            ]
            return jsonify({"results": results, "count": len(results)})
        if db_type == "graph":
            result = graph_engine.query_graph(components["graph_llm"], query, tenant_id)
            return jsonify({"answer": result.get("result", ""), "cypher": result.get("intermediate_steps", [])})
        return jsonify({"error": "Invalid type. Use 'vector' or 'graph'"}), 400
    except Exception as e:
        logging.error(f"Search error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/clear-history", methods=["POST"])
def clear_history_endpoint():
    tenant_id = _resolve_tenant()
    session_id = (request.json or {}).get("session_id")
    store.clear_session(tenant_id, session_id)
    return jsonify({"status": "success"}), 200


@app.route("/history", methods=["GET"])
def get_history_endpoint():
    tenant_id = _resolve_tenant()
    session_id = request.args.get("session_id", request.remote_addr)
    return jsonify({"session_id": session_id, "history": store.get_turns(tenant_id, session_id)})


@app.route("/stats", methods=["GET"])
def stats_endpoint():
    tenant_id = _resolve_tenant()
    components = tenant_registry.get_tenant(tenant_id)
    try:
        all_docs = components["vectorstore"]._collection.get(include=["metadatas"])
        vector_count = len(all_docs["ids"])
        unique_files = len({m.get("source", "") for m in all_docs["metadatas"] if m})
        indexing = store.get_indexing_status(tenant_id)
        result = {
            "tenant_id": tenant_id,
            "vector_store": {"total_chunks": vector_count, "unique_files": unique_files},
            "indexing_queue": {
                s: sum(1 for v in indexing.values() if v["status"] == s)
                for s in ("queued", "processing", "completed", "failed")
            },
            "conversation_sessions": len(store.list_sessions(tenant_id)),
        }
        if Config.ENABLE_GRAPH:
            result["graph_store"] = graph_engine.stats(tenant_id)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error getting stats: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/export-graph", methods=["GET"])
def export_graph_endpoint():
    tenant_id = _resolve_tenant()
    if not Config.ENABLE_GRAPH:
        return jsonify({"error": "Graph engine disabled"}), 400
    try:
        graph = graph_engine.get_graph()
        nodes = graph.query(
            "MATCH (n {tenant_id: $t}) RETURN id(n) AS id, labels(n) AS labels, properties(n) AS properties",
            params={"t": tenant_id},
        )
        rels = graph.query(
            "MATCH (a {tenant_id: $t})-[r]->(b {tenant_id: $t}) "
            "RETURN id(a) AS source, id(b) AS target, type(r) AS type, properties(r) AS properties",
            params={"t": tenant_id},
        )
        return jsonify({"nodes": nodes, "relationships": rels})
    except Exception as e:
        logging.error(f"Export error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/debug-graph", methods=["GET"])
def debug_graph_endpoint():
    tenant_id = _resolve_tenant()
    if not Config.ENABLE_GRAPH:
        return jsonify({"error": "Graph engine disabled"}), 400
    try:
        graph = graph_engine.get_graph()
        node_stats = graph.query(
            "MATCH (n {tenant_id: $t}) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC",
            params={"t": tenant_id},
        )
        sample = graph.query(
            "MATCH (n {tenant_id: $t}) RETURN labels(n) AS labels, properties(n) AS properties LIMIT 10",
            params={"t": tenant_id},
        )
        return jsonify({"node_stats": node_stats, "sample_nodes": sample})
    except Exception as e:
        logging.error(f"Debug-graph error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/tenants", methods=["GET"])
def tenants_endpoint():
    return jsonify({"active_tenants": tenant_registry.list_tenants()})


@app.route("/health", methods=["GET"])
def health_endpoint():
    health = {"status": "healthy", "components": {}}

    try:
        components = tenant_registry.get_tenant(Config.DEFAULT_TENANT)
        components["vectorstore"]._collection.count()
        health["components"]["vector_store"] = "operational"
    except Exception as e:
        health["components"]["vector_store"] = f"error: {e}"
        health["status"] = "degraded"

    if Config.ENABLE_GRAPH:
        try:
            graph_engine.get_graph().query("RETURN 1")
            health["components"]["graph_store"] = "operational"
        except Exception as e:
            health["components"]["graph_store"] = f"error: {e}"
            health["status"] = "degraded"

    try:
        get_queue().connection.ping()
        health["components"]["redis_queue"] = "operational"
    except Exception as e:
        health["components"]["redis_queue"] = f"error: {e}"
        health["status"] = "degraded"

    try:
        tenant_registry.get_tenant(Config.DEFAULT_TENANT)["main_llm"].invoke("ping")
        health["components"]["llm"] = "operational"
    except Exception as e:
        health["components"]["llm"] = f"error: {e}"
        health["status"] = "degraded"

    status_code = 200 if health["status"] == "healthy" else 503
    return jsonify(health), status_code


if __name__ == "__main__":
    logging.info(f"Starting Flask server on http://{Config.HOST}:{Config.PORT}")
    logging.info(f"LLM provider: {Config.LLM_PROVIDER} ({Config.LLM_MODEL})")
    logging.info(f"Graph engine: {'enabled' if Config.ENABLE_GRAPH else 'disabled'}")
    from waitress import serve

    serve(app, host=Config.HOST, port=Config.PORT)
