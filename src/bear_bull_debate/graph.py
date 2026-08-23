from typing import Any

from langgraph.graph import START, StateGraph

from .config import Settings
from .llm import make_llm
from .nodes import make_judge_node, make_researcher_node, make_summarize_node
from .router import make_route_after_bull, make_route_after_summarize
from .state import DebateState
from .tools import TOOLS


def build_graph(
    settings: Settings,
    checkpointer: Any = None,
    models: dict[str, Any] | None = None,
    interrupt_before: list[str] | None = None,
):
    """Assemble the debate StateGraph."""
    models = models or {}
    bear_llm = (
        models.get("bear")
        or make_llm(settings.bear_model, base_url=settings.base_url).bind_tools(TOOLS)
    )
    bull_llm = (
        models.get("bull")
        or make_llm(settings.bull_model, base_url=settings.base_url).bind_tools(TOOLS)
    )
    judge_llm = models.get("judge") or make_llm(
        settings.judge_model, base_url=settings.base_url
    )
    summary_llm = models.get("summary") or make_llm(
        settings.summary_model, base_url=settings.base_url
    )

    builder = StateGraph(DebateState)
    builder.add_node("bear", make_researcher_node("bear", bear_llm, TOOLS, settings))
    builder.add_node("bull", make_researcher_node("bull", bull_llm, TOOLS, settings))
    builder.add_node("summarize", make_summarize_node(summary_llm, settings))
    builder.add_node("judge", make_judge_node(judge_llm))

    builder.add_edge(START, "bear")
    builder.add_edge("bear", "bull")
    builder.add_conditional_edges(
        "bull",
        make_route_after_bull(settings),
        {"summarize": "summarize", "judge": "judge", "bear": "bear"},
    )
    builder.add_conditional_edges(
        "summarize",
        make_route_after_summarize(settings),
        {"judge": "judge", "bear": "bear"},
    )

    return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
