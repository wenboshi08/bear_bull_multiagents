from langchain_core.messages import AIMessage

from bear_bull_debate import runner


class StubGraph:
    """Minimal graph stand-in returning a fixed result."""

    def __init__(self, result):
        self.result = result

    async def ainvoke(self, state, config):
        self.last_state = state
        self.last_config = config
        return self.result


def _stub_graph(monkeypatch):
    stub = StubGraph(
        {
            "messages": [AIMessage(content="FINAL REPORT")],
            "tool_outputs": ["[get_stock_price] AAPL latest price: $123.45 (mock)"],
        }
    )
    monkeypatch.setattr(
        runner, "build_graph", lambda settings, checkpointer=None: stub
    )
    return stub


def test_run_debate_sync_returns_report(monkeypatch, settings):
    stub = _stub_graph(monkeypatch)
    out = runner.run_debate("AAPL", max_rounds=2, settings=settings)
    assert out["final_report"] == "FINAL REPORT"
    assert out["tool_logs"] == ["[get_stock_price] AAPL latest price: $123.45 (mock)"]
    assert out["thread_id"]
    # The seed message carries the requested company.
    assert stub.last_state["company"] == "AAPL"
    assert stub.last_state["max_rounds"] == 2


async def test_run_debate_async_returns_report(monkeypatch, settings):
    stub = _stub_graph(monkeypatch)
    out = await runner.run_debate_async("AAPL", max_rounds=3, settings=settings)
    assert out["final_report"] == "FINAL REPORT"
    assert stub.last_state["max_rounds"] == 3


def test_run_debate_falls_back_to_settings_max_rounds(monkeypatch, settings):
    stub = _stub_graph(monkeypatch)
    out = runner.run_debate("AAPL", settings=settings)
    assert out["final_report"] == "FINAL REPORT"
    # settings.max_rounds (default 2) is used when max_rounds is not passed.
    assert stub.last_state["max_rounds"] == 2
