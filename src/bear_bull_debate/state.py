import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class DebateState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    round: int
    company: str
    summary: str
    tool_outputs: Annotated[list[str], operator.add]
    max_rounds: int
