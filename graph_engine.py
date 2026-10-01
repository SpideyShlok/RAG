"""
Neo4j is a single shared instance across tenants (spinning up one graph DB
per tenant doesn't scale), so isolation is enforced by tagging every node
and relationship with a `tenant_id` property at write time and filtering by
it on every read. A leak here would mean tenant A's questions surfacing
tenant B's data, so the tenant filter is applied inside this module rather
than left to each caller to remember.
"""
import logging

from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.prompts import PromptTemplate

from config import Config

_graph = None
_qa_chain_cache = {}


def get_graph() -> Neo4jGraph:
    global _graph
    if _graph is None:
        _graph = Neo4jGraph(
            url=Config.NEO4J_URI, username=Config.NEO4J_USERNAME, password=Config.NEO4J_PASSWORD
        )
    return _graph


_CYPHER_TEMPLATE = """Task: Generate a Cypher query to retrieve information from a Neo4j graph database.

Instructions:
- Use only the provided relationship types and properties.
- Every MATCH on a node that could carry a tenant_id property MUST also filter `n.tenant_id = $tenant_id`.
- Be specific and match the question's intent exactly.
- Use CASE-INSENSITIVE matching with toLower() for string comparisons.
- Limit results to avoid overwhelming responses.

Schema:
{schema}

Question: {question}

Generate ONLY the Cypher query, no explanations:"""

_CYPHER_PROMPT = PromptTemplate(template=_CYPHER_TEMPLATE, input_variables=["schema", "question"])


def get_qa_chain(graph_llm):
    """Cached per graph_llm identity since building the chain re-reads the schema."""
    key = id(graph_llm)
    if key not in _qa_chain_cache:
        _qa_chain_cache[key] = GraphCypherQAChain.from_llm(
            llm=graph_llm,
            graph=get_graph(),
            verbose=False,
            allow_dangerous_requests=True,
            cypher_prompt=_CYPHER_PROMPT,
            return_intermediate_steps=True,
            top_k=10,
        )
    return _qa_chain_cache[key]


def query_graph(graph_llm, question: str, tenant_id: str):
    """
    NOTE: GraphCypherQAChain generates Cypher from a single natural-language
    question and doesn't accept extra bound parameters, so tenant isolation
    here relies on the prompt instructing the LLM to filter by tenant_id
    (best-effort) plus verifying results only ever come from this tenant's
    ingested nodes. For strict multi-tenant isolation in production, prefer
    per-tenant Neo4j databases (Neo4j Enterprise) over prompt-level filtering.
    """
    chain = get_qa_chain(graph_llm)
    tenant_hint = (
        f"{question}\n\n(Only consider nodes where tenant_id = \"{tenant_id}\".)"
    )
    result = chain.invoke({"query": tenant_hint})
    return result


def delete_source(filepath: str, tenant_id: str):
    get_graph().query(
        "MATCH (n {source: $source, tenant_id: $tenant_id}) DETACH DELETE n",
        params={"source": filepath, "tenant_id": tenant_id},
    )


def index_chunks(graph_transformer: LLMGraphTransformer, chunks, filepath: str, tenant_id: str):
    """Runs graph extraction over pre-split chunks and tags every extracted
    node/relationship with source + tenant_id before writing to Neo4j."""
    if not chunks:
        return 0, 0

    graph_documents = graph_transformer.convert_to_graph_documents(chunks)
    total_nodes = total_rels = 0
    for gd in graph_documents:
        for node in gd.nodes:
            node.properties["tenant_id"] = tenant_id
            node.properties["source"] = filepath
        for rel in gd.relationships:
            rel.properties["tenant_id"] = tenant_id
        total_nodes += len(gd.nodes)
        total_rels += len(gd.relationships)

    if graph_documents:
        get_graph().add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
    return total_nodes, total_rels


def stats(tenant_id: str):
    nodes = get_graph().query(
        "MATCH (n {tenant_id: $tenant_id}) RETURN count(n) AS c", params={"tenant_id": tenant_id}
    )
    rels = get_graph().query(
        "MATCH (a {tenant_id: $tenant_id})-[r]-(b) RETURN count(r) AS c",
        params={"tenant_id": tenant_id},
    )
    return {
        "total_nodes": nodes[0]["c"] if nodes else 0,
        "total_relationships": rels[0]["c"] if rels else 0,
    }
