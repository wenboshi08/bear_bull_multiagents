from langchain_core.tools import ToolException, tool


@tool
def get_stock_price(company: str) -> str:
    """Get the latest mock stock price for a company."""
    company = (company or "").strip()
    if not company:
        raise ToolException("company parameter must not be empty")
    return f"{company} latest price: $123.45 (mock)"


@tool
def get_financials(company: str) -> str:
    """Get mock financial metrics (revenue, EPS, YoY growth) for a company."""
    company = (company or "").strip()
    if not company:
        raise ToolException("company parameter must not be empty")
    return f"{company} FY revenue: $10.0B, EPS: $3.20, YoY growth: +8% (mock)"


@tool
def get_news_sentiment(company: str) -> str:
    """Get mock recent news sentiment for a company."""
    company = (company or "").strip()
    if not company:
        raise ToolException("company parameter must not be empty")
    return f"{company} recent news sentiment: slightly negative (mock)"


TOOLS = [get_stock_price, get_financials, get_news_sentiment]
