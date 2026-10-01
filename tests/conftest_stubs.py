"""Test-only: inject fake modules for heavy deps we can't install/run here
(torch, chromadb, neo4j, real LLM backends) so we can still import and
exercise our own application code for bugs."""
import sys
import types
from unittest.mock import MagicMock


def _mod(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


# --- langchain_chroma ---
m = _mod("langchain_chroma")
class FakeChroma:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._collection = MagicMock()
        self._collection.get.return_value = {"ids": [], "metadatas": []}
    def delete(self, **kwargs): pass
m.Chroma = FakeChroma

# --- langchain_huggingface ---
m = _mod("langchain_huggingface")
m.HuggingFaceEmbeddings = MagicMock

# --- langchain_community (package + submodules) ---
pkg = _mod("langchain_community")
pkg.__path__ = []

m = _mod("langchain_community.cross_encoders")
class FakeCrossEncoder:
    def __init__(self, **kwargs): pass
    def score(self, pairs): return [0.5] * len(pairs)
m.HuggingFaceCrossEncoder = FakeCrossEncoder

m = _mod("langchain_community.graphs")
class FakeNeo4jGraph:
    def __init__(self, **kwargs): pass
    def query(self, *a, **k): return []
    def add_graph_documents(self, *a, **k): pass
m.Neo4jGraph = FakeNeo4jGraph

m = _mod("langchain_community.storage")
class FakeRedisStore:
    def __init__(self, **kwargs): pass
m.RedisStore = FakeRedisStore

m = _mod("langchain_community.document_loaders")
class _FakeLoader:
    def __init__(self, *a, **k): pass
    def load(self): return []
m.PyPDFLoader = _FakeLoader
m.UnstructuredFileLoader = _FakeLoader
m.WebBaseLoader = _FakeLoader

# --- langchain (package + submodules used) ---
pkg = _mod("langchain")
pkg.__path__ = []

m = _mod("langchain.retrievers")
class FakeParentDocumentRetriever:
    def __init__(self, **kwargs):
        self.docstore = MagicMock()
    def add_documents(self, *a, **k): pass
    def invoke(self, q): return []
class FakeMultiQueryRetriever:
    @classmethod
    def from_llm(cls, **kwargs):
        inst = cls()
        return inst
    def invoke(self, q): return []
m.ParentDocumentRetriever = FakeParentDocumentRetriever
m.MultiQueryRetriever = FakeMultiQueryRetriever

m = _mod("langchain.retrievers.contextual_compression")
m.ContextualCompressionRetriever = MagicMock

m = _mod("langchain.retrievers.document_compressors.cross_encoder_rerank")
m.CrossEncoderReranker = MagicMock

m = _mod("langchain.storage")
class FakeInMemoryStore:
    def __init__(self, **kwargs): pass
m.InMemoryStore = FakeInMemoryStore

m = _mod("langchain.text_splitter")
class FakeSplitter:
    def __init__(self, **kwargs): pass
    def split_documents(self, docs): return docs
m.RecursiveCharacterTextSplitter = FakeSplitter

m = _mod("langchain.chains")
class FakeGraphCypherQAChain:
    @classmethod
    def from_llm(cls, **kwargs):
        return cls()
    def invoke(self, q): return {"result": "fake graph answer", "query": "MATCH (n) RETURN n", "intermediate_steps": []}
m.GraphCypherQAChain = FakeGraphCypherQAChain

# --- langchain_experimental ---
pkg = _mod("langchain_experimental")
pkg.__path__ = []
m = _mod("langchain_experimental.graph_transformers")
class FakeLLMGraphTransformer:
    def __init__(self, **kwargs): pass
    def convert_to_graph_documents(self, chunks):
        return []
m.LLMGraphTransformer = FakeLLMGraphTransformer

# --- LLM backends ---
class FakeChatModel:
    def __init__(self, **kwargs): pass
    def invoke(self, prompt):
        class R:
            content = "vector"
        return R()
    def stream(self, prompt_or_messages):
        yield "Hello "
        yield "world."

m = _mod("langchain_ollama")
m.ChatOllama = FakeChatModel
m.OllamaLLM = FakeChatModel

m = _mod("langchain_openai")
m.ChatOpenAI = FakeChatModel

m = _mod("langchain_anthropic")
m.ChatAnthropic = FakeChatModel

m = _mod("langchain_unstructured")
m.UnstructuredLoader = _FakeLoader
