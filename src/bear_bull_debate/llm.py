import os

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


def make_llm(
    model: str, temperature: float = 0.0, base_url: str | None = None
) -> ChatOpenAI:
    """Create a ChatOpenAI instance for the given model name.

    The endpoint resolves as: explicit ``base_url`` argument > ``OPENAI_BASE_URL``
    env var > DeepSeek's public endpoint (``https://api.deepseek.com``). The API key
    is taken from ``OPENAI_API_KEY``. This makes OpenAI-compatible providers
    (DeepSeek / Qwen) work in Colab, scripts, and the FastAPI server without
    relying on ``ChatOpenAI``'s implicit env-var probing.

    DeepSeek v4 models default to *thinking mode*; when ``tools`` are attached,
    the model's ``reasoning_content`` must be passed back in every follow-up
    request, but ``ChatOpenAI`` neither extracts nor round-trips that field, which
    makes the API reject the turn with HTTP 400. The debate only needs the final
    argument, so we disable thinking for DeepSeek endpoints via ``extra_body``.
    """
    kwargs = {"model": model, "temperature": temperature}
    resolved_base_url = base_url or os.getenv(
        "OPENAI_BASE_URL", "https://api.deepseek.com"
    )
    api_key = os.getenv("OPENAI_API_KEY")
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    if "deepseek" in (resolved_base_url or ""):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    if api_key:
        kwargs["api_key"] = api_key
    if os.getenv("VERBOSE") == "1":
        from langchain_core.callbacks import ConsoleCallbackHandler

        kwargs["callbacks"] = [ConsoleCallbackHandler()]
    return ChatOpenAI(**kwargs)


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
