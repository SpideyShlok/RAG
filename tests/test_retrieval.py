from langchain_core.documents import Document as LC_Doc
import retrieval as r


def test_rrf_fusion_dedupes_and_prioritizes_agreement():
    vec_docs = [
        LC_Doc(page_content="Flower Ltd budget report 2023", metadata={"source": "a.pdf"}),
        LC_Doc(page_content="Company overview", metadata={"source": "b.pdf"}),
    ]
    bm25_docs = [
        LC_Doc(page_content="Flower Ltd budget report 2023", metadata={"source": "a.pdf"}),
        LC_Doc(page_content="Shlok CS degree", metadata={"source": "c.pdf"}),
    ]
    fused = r.reciprocal_rank_fusion([vec_docs, bm25_docs], weights=[0.6, 0.4])
    assert len(fused) == 3, "the doc found by both retrievers should be deduplicated, not doubled"
    assert fused[0].metadata["source"] == "a.pdf", "doc found by both retrievers should rank first"


class _FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores

    def score(self, pairs):
        return self.scores[: len(pairs)]


def test_grounding_gate_keeps_relevant_and_drops_noise():
    docs = [LC_Doc(page_content="relevant", metadata={}), LC_Doc(page_content="noise", metadata={})]
    scored = r.score_with_cross_encoder(_FakeCrossEncoder([0.9, 0.05]), "q", docs)
    grounded, kept = r.grounding_gate(scored, min_score=0.15, top_n=5)
    assert grounded is True
    assert len(kept) == 1
    assert kept[0].page_content == "relevant"


def test_grounding_gate_refuses_when_everything_is_noise():
    docs = [LC_Doc(page_content="noise1", metadata={}), LC_Doc(page_content="noise2", metadata={})]
    scored = r.score_with_cross_encoder(_FakeCrossEncoder([0.01, 0.02]), "q", docs)
    grounded, kept = r.grounding_gate(scored)
    assert grounded is False
    assert kept == []


def test_router_parses_unambiguous_output():
    assert r.parse_router_output("Choice: GRAPH") == "graph"
    assert r.parse_router_output("vector") == "vector"


def test_router_flags_ambiguous_output():
    assert r.parse_router_output("not sure, could be either") == "ambiguous"


def test_resolve_route_falls_back_to_hybrid_on_ambiguity():
    route, used_fallback = r.resolve_route("graph or vector, hard to say")
    assert route == "hybrid"
    assert used_fallback is True


def test_resolve_route_trusts_unambiguous_output():
    route, used_fallback = r.resolve_route("GRAPH")
    assert route == "graph"
    assert used_fallback is False


class _FakeMsg:
    def __init__(self, msg_type, content):
        self.type = msg_type
        self.content = content


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        from types import SimpleNamespace
        return SimpleNamespace(content=self.reply)


def test_query_rewriting_pulls_context_from_history():
    history = [_FakeMsg("human", "Tell me about Flower Ltd"), _FakeMsg("ai", "It's a logistics company.")]
    llm = _FakeLLM("What is the revenue of Flower Ltd?")
    out = r.rewrite_query_with_history(llm, "what about their revenue?", history)
    assert out == "What is the revenue of Flower Ltd?"
    assert "Flower Ltd" in llm.last_prompt


def test_query_rewriting_skips_llm_call_with_no_history():
    class ExplodingLLM:
        def invoke(self, prompt):
            raise AssertionError("should not be called when there is no history to rewrite against")

    out = r.rewrite_query_with_history(ExplodingLLM(), "hello", [])
    assert out == "hello"
