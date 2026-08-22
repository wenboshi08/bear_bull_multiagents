import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from bear_bull_debate.api import DebateRequest, create_app, run_debate


class StubGraph:
    def __init__(self, result):
        self.result = result

    async def ainvoke(self, state, config):
        self.last_state = state
        self.last_config = config
        return self.result


def stub_graph():
    return StubGraph(
        {"messages": [AIMessage(content="FINAL REPORT")], "tool_outputs": []}
    )


async def test_run_debate_returns_final_report():
    graph = stub_graph()
    req = DebateRequest(company="AAPL", thread_id="t-1", max_rounds=2)
    out = await run_debate(graph, req)
    assert out["final_report"] == "FINAL REPORT"
    assert out["thread_id"] == "t-1"
    assert out["tool_logs"] == []


def test_debate_request_rejects_invalid_company():
    with pytest.raises(ValidationError):
        DebateRequest(company="BAD; rm -rf /")
    with pytest.raises(ValidationError):
        DebateRequest(company="")


def test_debate_request_rejects_out_of_range_rounds():
    with pytest.raises(ValidationError):
        DebateRequest(company="AAPL", max_rounds=9)


def test_debate_request_accepts_valid_input():
    req = DebateRequest(company="AAPL", max_rounds=3)
    assert req.max_rounds == 3
    assert req.thread_id


def test_api_endpoint_smoke():
    app = create_app(graph=stub_graph())
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/debate", json={"company": "AAPL", "max_rounds": 2}
        )
    assert resp.status_code == 200
    assert resp.json()["final_report"] == "FINAL REPORT"


def test_api_endpoint_rejects_invalid_company():
    app = create_app(graph=stub_graph())
    with TestClient(app) as client:
        resp = client.post("/api/v1/debate", json={"company": "BAD; DROP TABLE"})
    assert resp.status_code == 422


def test_healthz():
    app = create_app(graph=stub_graph())
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
