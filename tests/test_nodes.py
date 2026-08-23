from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

from bear_bull_debate.config import Settings
from bear_bull_debate.nodes import (
    _merge_consecutive_assistant_messages,
    _strip_orphaned_tool_messages,
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


async def test_researcher_caps_tool_loop():
    settings = Settings(max_tool_rounds=2, checkpointer_uri=None)
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_stock_price",
                    "args": {"company": "AAPL"},
                    "id": f"call_{i}",
                    "type": "tool_call",
                }
            ],
        )
        for i in range(10)
    ]
    llm = FakeMessagesListChatModel(responses=responses)
    node = make_researcher_node("bear", llm, TOOLS, settings)
    result = await node(make_state())

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2


async def test_researcher_cap_forces_final_argument():
    settings = Settings(max_tool_rounds=2, checkpointer_uri=None)
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_stock_price",
                    "args": {"company": "AAPL"},
                    "id": f"call_{i}",
                    "type": "tool_call",
                }
            ],
        )
        for i in range(2)
    ]
    # After the loop exhausts, the node forces one final synthesis call.
    responses.append(AIMessage(content="Final bear thesis (synthesized)"))
    llm = FakeMessagesListChatModel(responses=responses)
    node = make_researcher_node("bear", llm, TOOLS, settings)
    result = await node(make_state())

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2
    # The turn must end with an argument, not a dangling ToolMessage.
    assert result["messages"][-1].content == "Final bear thesis (synthesized)"


async def test_researcher_handles_malformed_tool_call(settings):
    from types import SimpleNamespace

    class _MalformedLLM:
        def __init__(self, follow_up):
            self._follow_up = follow_up
            self._calls = 0

        async def ainvoke(self, messages):
            self._calls += 1
            if self._calls == 1:
                return SimpleNamespace(
                    content="", tool_calls=[{"args": {"company": "AAPL"}}]
                )
            return self._follow_up

    llm = _MalformedLLM(follow_up=AIMessage(content="Recovered from malformed tool call."))
    node = make_researcher_node("bear", llm, TOOLS, settings)
    result = await node(make_state())

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Unknown tool" in tool_messages[0].content
    assert tool_messages[0].name == "unknown"
    assert result["messages"][-1].content == "Recovered from malformed tool call."


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


def test_strip_orphaned_tool_messages():
    """Defense-in-depth: ToolMessages without a preceding tool_calls AIMessage
    are dropped before the list reaches the API (which would 400 otherwise)."""
    tool_call_ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "get_stock_price", "args": {"company": "AAPL"}, "id": "c1", "type": "tool_call"}
        ],
        id="tc",
    )
    paired_tool = ToolMessage(content="price", tool_call_id="c1", name="get_stock_price", id="t1")
    final_arg = AIMessage(content="arg", id="arg")
    orphan_1 = ToolMessage(content="orphan before any tool_calls", tool_call_id="cX", name="x", id="o1")
    orphan_2 = ToolMessage(content="orphan after final arg", tool_call_id="cY", name="x", id="o2")

    messages = [
        SystemMessage(content="sys", id="sys"),
        orphan_1,
        tool_call_ai,
        paired_tool,
        final_arg,
        orphan_2,
    ]
    sanitized = _strip_orphaned_tool_messages(messages)
    assert [m.id for m in sanitized] == ["sys", "tc", "t1", "arg"]


def test_merge_consecutive_assistant_messages():
    """DeepSeek v4 Flash rejects two consecutive assistant messages (HTTP 400
    'tool must be a response to a preceding message with tool_calls'); the
    merge collapses them while preserving tool pairing."""
    tc1 = AIMessage(
        content="",
        tool_calls=[
            {"name": "get_stock_price", "args": {"company": "AAPL"}, "id": "call_1", "type": "tool_call"}
        ],
        id="tc1",
    )
    tool1 = ToolMessage(content="r1", tool_call_id="call_1", name="get_stock_price", id="t1")
    arg1 = AIMessage(content="Bear final", id="arg1")
    tc2 = AIMessage(
        content="",
        tool_calls=[
            {"name": "get_financials", "args": {"company": "AAPL"}, "id": "call_2", "type": "tool_call"}
        ],
        id="tc2",
    )
    tool2 = ToolMessage(content="r2", tool_call_id="call_2", name="get_financials", id="t2")
    arg2 = AIMessage(content="Bull final", id="arg2")

    messages = [
        SystemMessage(content="s", id="s"),
        HumanMessage(content="h", id="h"),
        tc1,
        tool1,
        arg1,   # end of turn 1's argument
        tc2,    # start of turn 2's tool call  ← consecutive assistant!
        tool2,
        arg2,
    ]
    merged = _merge_consecutive_assistant_messages(messages)

    types = [m.type for m in merged]
    for i in range(len(types) - 1):
        assert not (types[i] == "ai" and types[i + 1] == "ai"), types
    assert types == ["system", "human", "ai", "tool", "ai", "tool", "ai"]
    # the merged assistant keeps arg1's content AND tc2's tool_calls
    merged_assistant = merged[4]
    assert isinstance(merged_assistant, AIMessage)
    assert "Bear final" in merged_assistant.content
    assert merged_assistant.tool_calls[0]["id"] == "call_2"


async def test_researcher_no_consecutive_assistant_messages(settings):
    """History crossing a turn boundary contains two consecutive assistant
    messages (previous argument + next tool call); the researcher must merge
    them before the list reaches the LLM (DeepSeek v4 rejects alternation
    violations with HTTP 400)."""
    state_messages = [
        HumanMessage(content="Debate topic: AAPL", id="seed"),
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
            id="tc1",
        ),
        ToolMessage(content="r1", tool_call_id="call_1", name="get_stock_price", id="t1"),
        AIMessage(content="Bear final argument", id="arg1"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_financials",
                    "args": {"company": "AAPL"},
                    "id": "call_2",
                    "type": "tool_call",
                }
            ],
            id="tc2",
        ),
        ToolMessage(content="r2", tool_call_id="call_2", name="get_financials", id="t2"),
        AIMessage(content="Bull final argument", id="arg2"),
    ]
    llm = CapturingLLM(response=AIMessage(content="Next round argument"))
    node = make_researcher_node("bear", llm, TOOLS, settings)
    await node(make_state(messages=state_messages, round=1))

    seen_types = [m.type for m in llm.seen[0]]
    for i in range(len(seen_types) - 1):
        assert not (seen_types[i] == "ai" and seen_types[i + 1] == "ai"), seen_types


class CapturingLLM:
    """Fake LLM that records every message list it is called with."""

    def __init__(self, response):
        self.response = response
        self.seen = []

    async def ainvoke(self, messages):
        self.seen.append(list(messages))
        return self.response


async def test_researcher_window_keeps_tool_call_pairing():
    # State where the naive last-4 history slice starts with a ToolMessage
    # whose AIMessage(tool_calls) falls outside the window. _safe_window must
    # extend left so the LLM never sees an orphaned ToolMessage (which the
    # OpenAI/DeepSeek API rejects with HTTP 400).
    settings = Settings(history_window=4, checkpointer_uri=None)
    state_messages = [
        HumanMessage(content="Debate topic: AAPL", id="seed"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_stock_price",
                    "args": {"company": "AAPL"},
                    "id": "c1",
                    "type": "tool_call",
                }
            ],
            id="m_tc1",
        ),
        ToolMessage(
            content="AAPL latest price: $123.45 (mock)",
            tool_call_id="c1",
            name="get_stock_price",
            id="m_t1",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_financials",
                    "args": {"company": "AAPL"},
                    "id": "c2",
                    "type": "tool_call",
                }
            ],
            id="m_tc2",
        ),
        ToolMessage(
            content="AAPL FY revenue: $10.0B (mock)",
            tool_call_id="c2",
            name="get_financials",
            id="m_t2",
        ),
        AIMessage(content="Bear final argument", id="m_final"),
    ]
    llm = CapturingLLM(response=AIMessage(content="Bull rebuttal"))
    node = make_researcher_node("bull", llm, TOOLS, settings)
    await node(make_state(messages=state_messages, round=1))

    seen = llm.seen[0]
    # Skip system/seed prefix; the first history message must not be a ToolMessage.
    first = next(m for m in seen if m.type in ("ai", "tool", "human"))
    assert first.type != "tool"
    assert first.id == "m_tc1"  # window extended left to keep the pairing


async def test_summarize_keeps_tool_call_pairing():
    # The naive keep-window (last 2) would start with ToolMessage m_t1 whose
    # AIMessage(tool_calls) m_tc is in the removed prefix. _safe_window must
    # keep m_tc so the surviving messages stay protocol-valid.
    settings = Settings(history_window=2, checkpointer_uri=None)
    messages = [
        HumanMessage(content="Debate topic: AAPL", id="s1"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_stock_price",
                    "args": {"company": "AAPL"},
                    "id": "c1",
                    "type": "tool_call",
                }
            ],
            id="m_tc",
        ),
        ToolMessage(
            content="AAPL latest price: $123.45 (mock)",
            tool_call_id="c1",
            name="get_stock_price",
            id="m_t1",
        ),
        AIMessage(content="Bear final argument", id="m_arg"),
    ]
    llm = FakeMessagesListChatModel(responses=[AIMessage(content="summary")])
    node = make_summarize_node(llm, settings)
    result = await node(make_state(messages=messages, summary="prior"))

    removed_ids = {m.id for m in result["messages"] if isinstance(m, RemoveMessage)}
    # Only the seed is removed; m_t1 survives WITH its m_tc predecessor.
    assert removed_ids == {"s1"}
