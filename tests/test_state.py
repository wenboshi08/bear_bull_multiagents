from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from bear_bull_debate.state import DebateState


def test_add_messages_appends_and_dedups():
    m1 = HumanMessage(content="hi", id="1")
    m2 = AIMessage(content="hello", id="2")
    merged = add_messages([m1], [m2, m1])
    assert [m.content for m in merged] == ["hi", "hello"]


def test_remove_message_deletes_by_id():
    m1 = HumanMessage(content="a", id="1")
    m2 = AIMessage(content="b", id="2")
    result = add_messages([m1, m2], [RemoveMessage(id="1")])
    assert [m.content for m in result] == ["b"]


def test_tool_outputs_append_across_nodes():
    builder = StateGraph(DebateState)

    def node_a(state):
        return {"tool_outputs": ["A"]}

    def node_b(state):
        return {"tool_outputs": ["B"]}

    builder.add_node("a", node_a)
    builder.add_node("b", node_b)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    graph = builder.compile()

    result = graph.invoke(
        {
            "messages": [],
            "round": 0,
            "company": "X",
            "summary": "",
            "tool_outputs": ["seed"],
            "max_rounds": 2,
        }
    )
    # operator.add makes tool_outputs APPEND, not overwrite
    assert result["tool_outputs"] == ["seed", "A", "B"]
