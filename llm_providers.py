"""
Swap LLM backends via env vars instead of code changes.

LLM_PROVIDER=ollama            -> local Ollama server (LLM_API_BASE optional, default http://localhost:11434)
LLM_PROVIDER=openai            -> OpenAI API (needs LLM_API_KEY)
LLM_PROVIDER=openai_compatible -> any OpenAI-compatible endpoint (vLLM, LM Studio, Groq, etc.)
                                   needs LLM_API_BASE, LLM_API_KEY may be a dummy value
LLM_PROVIDER=anthropic         -> Claude models (needs LLM_API_KEY)
"""
import logging
from config import Config


class UnsupportedProviderError(ValueError):
    pass


def _make_llm(provider: str, model: str, api_base, api_key, temperature: float = 0.0):
    provider = (provider or "ollama").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        kwargs = {"model": model, "temperature": temperature}
        if api_base:
            kwargs["base_url"] = api_base
        return ChatOllama(**kwargs)

    if provider in ("openai", "openai_compatible"):
        from langchain_openai import ChatOpenAI
        kwargs = {"model": model, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        if provider == "openai_compatible":
            if not api_base:
                raise UnsupportedProviderError("LLM_API_BASE is required for openai_compatible provider")
            kwargs["base_url"] = api_base
        return ChatOpenAI(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        if not api_key:
            raise UnsupportedProviderError("LLM_API_KEY is required for anthropic provider")
        return ChatAnthropic(model=model, temperature=temperature, api_key=api_key)

    raise UnsupportedProviderError(
        f"Unknown LLM_PROVIDER '{provider}'. Use one of: ollama, openai, openai_compatible, anthropic"
    )


def get_main_llm(temperature: float = 0.0):
    """The primary LLM used for answering, routing, and query rewriting."""
    logging.info(f"Loading main LLM: provider={Config.LLM_PROVIDER} model={Config.LLM_MODEL}")
    return _make_llm(Config.LLM_PROVIDER, Config.LLM_MODEL, Config.LLM_API_BASE, Config.LLM_API_KEY, temperature)


def get_graph_llm(temperature: float = 0.0):
    """
    A (usually smaller/cheaper) LLM dedicated to graph extraction and Cypher
    generation, so a heavy main model doesn't bottleneck indexing throughput.
    """
    logging.info(f"Loading graph LLM: provider={Config.GRAPH_LLM_PROVIDER} model={Config.GRAPH_LLM_MODEL}")
    return _make_llm(
        Config.GRAPH_LLM_PROVIDER, Config.GRAPH_LLM_MODEL, Config.LLM_API_BASE, Config.LLM_API_KEY, temperature
    )
