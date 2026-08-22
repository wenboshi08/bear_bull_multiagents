from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from bear_bull_debate.nodes import (
    make_judge_node,
    make_researcher_node,
    make_summarize_node,
)
from bear_bull_debate.tools import TOOLS


def make_state(**overrides):
    state = {
        "messages": [HumanMessage(content="Debate topic: AAPL", id="seed")],
        "round": 0,
        "company": "AAPL",
        "summary": "",
        "tool_outputs": [],
        "max_rounds": 2,
    }
    state.update(overrides)
    return state


async def test_researcher_no_tool_calls(settings):
    llm = FakeMessagesListChatModel(
        responses=[AIMessage(content="Bear argument: AAPL overvalued")]
    )
    node = make_researcher_node("bear", llm, TOOLS, settings)
    result = await node(make_state())
    assert result["messages"][-1].content == "Bear argument: AAPL overvalued"
    assert "round" not in result


async def test_researcher_executes_tool_and_logs_output(settings):
    llm = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_stock_price",
                        "args": {"company": "AAPL"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="After checking price: bear thesis"),
        ]
    )
    node = make_researcher_node("bear", llm, TOOLS, settings)
    result = await node(make_state())

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call_1"
    assert "123.45" in tool_messages[0].content
    assert result["tool_outputs"] == [
        "[get_stock_price] AAPL latest price: $123.45 (mock)"
    ]


async def test_researcher_tool_exception_degrades_gracefully(settings):
    llm = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_stock_price",
                        "args": {"company": ""},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Tool failed, arguing from general knowledge."),
        ]
    )
    node = make_researcher_node("bear", llm, TOOLS, settings)
    result = await node(make_state())

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Tool error" in tool_messages[0].content
    assert (
        result["messages"][-1].content
        == "Tool failed, arguing from general knowledge."
    )


async def test_bull_increments_round(settings):
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="Bull argument")])
    node = make_researcher_node("bull", llm, TOOLS, settings)
    result = await node(make_state(round=1))
    assert result["round"] == 2


async def test_summarize_removes_old_messages(settings):
    old = [
        HumanMessage(content="Debate topic: AAPL", id="s1"),
        AIMessage(content="Bear R1", id="m1"),
        AIMessage(content="Bull R1", id="m2"),
        AIMessage(content="Bear R2", id="m3"),
        AIMessage(content="Bull R2", id="m4"),
    ]
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="Compressed summary")])
    node = make_summarize_node(llm, settings)
    result = await node(make_state(messages=old, summary="prior"))

    assert result["summary"] == "Compressed summary"
    removed_ids = {m.id for m in result["messages"] if isinstance(m, RemoveMessage)}
    # history_window=4 keeps the last 4 (m1..m4); only the seed "s1" is removed
    assert removed_ids == {"s1"}


async def test_summarize_noop_when_short(settings):
    short = [HumanMessage(content="hi", id="s1"), AIMessage(content="a", id="m1")]
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="should not be called")])
    node = make_summarize_node(llm, settings)
    result = await node(make_state(messages=short, summary="prior"))
    assert result["summary"] == "prior"


async def test_judge_appends_report(settings):
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="## Verdict\nNeutral")])
    node = make_judge_node(llm)
    result = await node(make_state())
    assert result["messages"][-1].content.startswith("## Verdict")
