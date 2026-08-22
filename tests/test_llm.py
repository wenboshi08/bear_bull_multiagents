import httpx
import pytest
from openai import APIConnectionError

from bear_bull_debate.llm import ainvoke_with_retry


class FlakyLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        raise APIConnectionError(
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        )


class ValueErrLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        raise ValueError("boom")


async def test_retries_transient_errors_three_times():
    llm = FlakyLLM()
    with pytest.raises(APIConnectionError):
        await ainvoke_with_retry(llm, [])
    assert llm.calls == 3


async def test_does_not_retry_non_transient_errors():
    llm = ValueErrLLM()
    with pytest.raises(ValueError):
        await ainvoke_with_retry(llm, [])
    assert llm.calls == 1
