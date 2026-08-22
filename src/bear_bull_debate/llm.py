from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

RETRYABLE_EXCEPTIONS = (APIConnectionError, APITimeoutError, RateLimitError)


def make_llm(model: str, temperature: float = 0.0) -> ChatOpenAI:
    """Create a ChatOpenAI instance for the given model name."""
    return ChatOpenAI(model=model, temperature=temperature)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
)
async def ainvoke_with_retry(
    llm: BaseChatModel, messages: list[BaseMessage]
) -> BaseMessage:
    """Invoke the LLM, retrying transient network/rate-limit errors up to 3 times."""
    return await llm.ainvoke(messages)
