from collections.abc import Callable
from typing import Any

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import ToolException

from .config import Settings
from .llm import ainvoke_with_retry
from .prompts import (
    BEAR_SYSTEM_PROMPT,
    BULL_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
)
from .state import DebateState
from .tracing import trace


def make_researcher_node(
    role: str, llm: Any, tools: list[Any], settings: Settings
) -> Callable:
    """Build a Bear or Bull researcher node with a tool-calling loop."""
    tools_by_name = {t.name: t for t in tools}
    system_tpl = BEAR_SYSTEM_PROMPT if role == "bear" else BULL_SYSTEM_PROMPT

    async def node(state: DebateState) -> dict:
        system = SystemMessage(content=system_tpl.format(company=state["company"]))
        history = list(state["messages"])[-settings.history_window :]
        seed: list[BaseMessage] = [system]
        if state["summary"]:
            seed.append(
                SystemMessage(content=f"Prior debate summary:\n{state['summary']}")
            )

        call_messages: list[BaseMessage] = seed + history
        new_messages: list[BaseMessage] = []
        tool_logs: list[str] = []

        while True:
            response = await ainvoke_with_retry(llm, call_messages)
            new_messages.append(response)
            call_messages.append(response)

            tool_calls = getattr(response, "tool_calls", None)
            if not tool_calls:
                break

            for tc in tool_calls:
                name = tc["name"]
                try:
                    tool = tools_by_name[name]
                    raw = await tool.ainvoke(tc["args"])
                    content = str(raw)
                    tool_logs.append(f"[{name}] {content}")
                except ToolException as exc:
                    content = f"Tool error: {exc}. Please adjust your parameters and retry."
                except KeyError:
                    content = f"Unknown tool '{name}'. Use only the provided tools."
                tool_msg = ToolMessage(
                    content=content, tool_call_id=tc["id"], name=name
                )
                new_messages.append(tool_msg)
                call_messages.append(tool_msg)

        result: dict = {"messages": new_messages}
        if role == "bull":
            result["round"] = state["round"] + 1
        if tool_logs:
            result["tool_outputs"] = tool_logs
        return result

    node = trace(f"{role}_researcher")(node)
    return node


def _format_messages(messages: list[BaseMessage]) -> str:
    return "\n".join(f"{type(m).__name__}: {m.content}" for m in messages)


def make_summarize_node(summary_llm: Any, settings: Settings) -> Callable:
    """Compacts old messages into a summary and frees them via RemoveMessage."""

    async def node(state: DebateState) -> dict:
        messages = list(state["messages"])
        if len(messages) <= settings.history_window:
            return {"summary": state["summary"]}

        old = messages[: -settings.history_window]

        user = (
            f"Previous summary:\n{state['summary'] or '(none)'}\n\n"
            f"New messages to summarize:\n{_format_messages(old)}"
        )
        resp = await ainvoke_with_retry(
            summary_llm,
            [SystemMessage(content=SUMMARY_SYSTEM_PROMPT), HumanMessage(content=user)],
        )

        remove = [RemoveMessage(id=m.id) for m in old if m.id]
        return {"summary": str(resp.content), "messages": remove}

    node = trace("summarize")(node)
    return node


def make_judge_node(judge_llm: Any) -> Callable:
    """Judge reads summary + recent messages and writes the final report."""

    async def node(state: DebateState) -> dict:
        context_parts: list[str] = []
        if state["summary"]:
            context_parts.append(f"### Earlier debate summary\n{state['summary']}")
        context_parts.append(f"### Recent debate\n{_format_messages(state['messages'])}")
        context_parts.append(f"### Company under debate\n{state['company']}")

        resp = await ainvoke_with_retry(
            judge_llm,
            [
                SystemMessage(
                    content=JUDGE_SYSTEM_PROMPT.format(company=state["company"])
                ),
                HumanMessage(content="\n\n".join(context_parts)),
            ],
        )
        return {"messages": [resp]}

    node = trace("judge")(node)
    return node
