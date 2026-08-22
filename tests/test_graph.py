import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from bear_bull_debate.checkpointer import make_checkpointer
from bear_bull_debate.config import Settings
from bear_bull_debate.graph import build_graph


def make_initial(company="AAPL", max_rounds=2):
    return {
        "company": company,
        "round": 0,
        "max_rounds": max_rounds,
        "messages": [HumanMessage(content=f"Debate topic: {company}", id="seed")],
        "summary": "",
        "tool_outputs": [],
    }


def make_models(rounds=2):
    return {
        "bear": FakeMessagesListChatModel(
            responses=[AIMessage(content=f"Bear R{i}") for i in range(1, rounds + 1)]
        ),
        "bull": FakeMessagesListChatModel(
            responses=[AIMessage(content=f"Bull R{i}") for i in range(1, rounds + 1)]
        ),
        "judge": FakeMessagesListChatModel(responses=[AIMessage(content="FINAL REPORT")]),
        "summary": FakeMessagesListChatModel(responses=[AIMessage(content="SUMMARY")]),
    }


async def test_graph_runs_two_rounds_then_judge(settings):
    graph = build_graph(settings, checkpointer=None, models=make_models(rounds=2))
    result = await graph.ainvoke(
        make_initial(), config={"configurable": {"thread_id": "t1"}}
    )

    assert result["round"] == 2
    assert result["messages"][-1].content == "FINAL REPORT"
    assert result["tool_outputs"] == []


async def test_graph_summarizes_when_threshold_exceeded():
    settings = Settings(
        max_rounds=3, history_window=2, message_threshold=4, checkpointer_uri=None
    )
    graph = build_graph(settings, checkpointer=None, models=make_models(rounds=3))
    result = await graph.ainvoke(
        make_initial(max_rounds=3), config={"configurable": {"thread_id": "t2"}}
    )

    assert result["summary"] == "SUMMARY"
    assert result["round"] == 3
    assert result["messages"][-1].content == "FINAL REPORT"
    # summarize compacted the early messages
    assert len(result["messages"]) < 7


async def test_resume_from_interrupt_before_judge(settings):
    from langgraph.checkpoint.memory import InMemorySaver

    graph = build_graph(
        settings,
        checkpointer=InMemorySaver(),
        models=make_models(rounds=2),
        interrupt_before=["judge"],
    )
    config = {"configurable": {"thread_id": "thread-2"}}
    first = await graph.ainvoke(make_initial(), config)
    assert first["round"] == 2

    final = await graph.ainvoke(None, config)
    assert final["messages"][-1].content == "FINAL REPORT"


async def test_checkpointer_factory_memory():
    cp = await make_checkpointer(None)
    assert cp is not None


async def test_checkpointer_factory_postgres_missing_extra(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres.aio", None)
    with pytest.raises(RuntimeError, match="postgres"):
        await make_checkpointer("postgresql://localhost/db")
