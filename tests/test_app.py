from unittest.mock import MagicMock, patch

from langchain_core.documents import Document as LC_Doc
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import app
import store


def _fake_components(llm, retriever, cross_encoder):
    vs = MagicMock()
    vs._collection.get.return_value = {"ids": [], "metadatas": []}
    return {
        "tenant_id": "default",
        "vectorstore": vs,
        "main_llm": llm,
        "graph_llm": llm,
        "multi_query_retriever": retriever,
        "cross_encoder": cross_encoder,
    }


# ------------------------------------------------------- tenant resolution --

def test_tenant_header_wins_on_non_json_request():
    with app.app.test_request_context("/check-files", method="GET", headers={"X-Tenant-ID": "acme"}):
        assert app._resolve_tenant() == "acme"


def test_tenant_defaults_when_nothing_given():
    with app.app.test_request_context("/check-files", method="GET"):
        assert app._resolve_tenant() == "default"


def test_tenant_from_json_body():
    with app.app.test_request_context("/ask", method="POST", json={"prompt": "hi", "tenant": "globex"}):
        assert app._resolve_tenant() == "globex"


def test_tenant_from_query_arg():
    with app.app.test_request_context("/check-files?tenant=initech", method="GET"):
        assert app._resolve_tenant() == "initech"


# ------------------------------------------------------------- /ask --------

def test_ask_grounded_answer_includes_citation():
    llm = FakeListChatModel(responses=["vector", "The answer is 42."])
    retriever = MagicMock()
    retriever.invoke.return_value = [
        LC_Doc(page_content="Flower Ltd revenue was 5 million in 2023.", metadata={"source": "a.pdf", "page": 1})
    ]
    ce = MagicMock()
    ce.score.return_value = [0.8]
    components = _fake_components(llm, retriever, ce)

    with patch("app.tenant_registry.get_tenant", return_value=components):
        client = app.app.test_client()
        store.clear_session("default", "grounded_test")
        resp = client.post("/ask", json={"prompt": "revenue?"}, headers={"X-Session-ID": "grounded_test"})
        body = resp.get_data(as_text=True)
        assert resp.status_code == 200
        assert "a.pdf" in body
        assert "[Retrieved from: vector]" in body


def test_ask_refuses_when_nothing_is_grounded():
    llm = FakeListChatModel(responses=["vector"])
    retriever = MagicMock()
    retriever.invoke.return_value = [LC_Doc(page_content="unrelated gardening text", metadata={"source": "b.pdf"})]
    ce = MagicMock()
    ce.score.return_value = [0.01]
    components = _fake_components(llm, retriever, ce)

    with patch("app.tenant_registry.get_tenant", return_value=components):
        client = app.app.test_client()
        store.clear_session("default", "refusal_test")
        resp = client.post("/ask", json={"prompt": "unanswerable"}, headers={"X-Session-ID": "refusal_test"})
        assert "don't have enough relevant, grounded information" in resp.get_data(as_text=True)


def test_ask_rewrites_followup_using_history_before_retrieving():
    store.clear_session("default", "followup_test")
    store.append_turn("default", "followup_test", "user", "Tell me about Flower Ltd")
    store.append_turn("default", "followup_test", "assistant", "It is a logistics company.")

    llm = FakeListChatModel(
        responses=["What was the revenue of Flower Ltd, the logistics company?", "vector", "Five million."]
    )
    retriever = MagicMock()
    retriever.invoke.return_value = [LC_Doc(page_content="Flower Ltd made 5 million.", metadata={"source": "a.pdf"})]
    ce = MagicMock()
    ce.score.return_value = [0.9]
    components = _fake_components(llm, retriever, ce)

    with patch("app.tenant_registry.get_tenant", return_value=components):
        client = app.app.test_client()
        resp = client.post(
            "/ask", json={"prompt": "what about their revenue?"}, headers={"X-Session-ID": "followup_test"}
        )
        assert resp.status_code == 200
        query_sent = retriever.invoke.call_args[0][0]
        assert "Flower Ltd" in query_sent, "query rewriting should carry prior context into retrieval"


def test_ask_rejects_empty_prompt():
    client = app.app.test_client()
    resp = client.post("/ask", json={"prompt": "  "})
    assert resp.status_code == 400
