import pytest
from langchain_core.tools import ToolException

from bear_bull_debate.tools import get_financials, get_stock_price, get_news_sentiment


def test_get_stock_price_returns_mock_value():
    result = get_stock_price.invoke({"company": "AAPL"})
    assert "AAPL" in result
    assert "mock" in result


def test_get_financials_returns_mock_value():
    result = get_financials.invoke({"company": "TSLA"})
    assert "TSLA" in result
    assert "EPS" in result


def test_get_news_sentiment_returns_mock_value():
    result = get_news_sentiment.invoke({"company": "NVDA"})
    assert "NVDA" in result


def test_tool_raises_tool_exception_on_empty_company():
    with pytest.raises(ToolException):
        get_stock_price.invoke({"company": ""})
