"""
LLM Provider Abstraction
Swap between OpenAI and Anthropic by changing LLM_PROVIDER in .env
"""
from src.config import (
    LLM_PROVIDER, OPENAI_API_KEY, ANTHROPIC_API_KEY,
    OPENAI_MODEL, ANTHROPIC_MODEL, EMBEDDING_MODEL
)


def get_chat_llm(temperature: float = 0.0):
    """Get the chat LLM based on configured provider."""
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=ANTHROPIC_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=temperature,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=temperature,
        )


def get_embeddings():
    """Get embedding model. Always uses OpenAI embeddings regardless of LLM provider."""
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )
