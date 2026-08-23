import httpx
import pytest
from openai import APIConnectionError

from bear_bull_debate.llm import ainvoke_with_retry, make_llm


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


def test_make_llm_injects_base_url_and_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.example.com")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    llm = make_llm("deepseek-v4-flash")
    assert llm.openai_api_base == "https://custom.example.com"
    assert llm.openai_api_key.get_secret_value() == "test-key"


def test_make_llm_defaults_to_deepseek_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    llm = make_llm("deepseek-v4-flash")
    assert llm.openai_api_base == "https://api.deepseek.com"
    assert llm.openai_api_key.get_secret_value() == "test-key"


def test_make_llm_base_url_param_overrides_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.example.com")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    llm = make_llm("deepseek-v4-flash", base_url="https://api.deepseek.com")
    assert llm.openai_api_base == "https://api.deepseek.com"
