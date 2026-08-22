from collections.abc import Callable

from .config import Settings
from .state import DebateState


def make_route_after_bull(settings: Settings) -> Callable[[DebateState], str]:
    def route(state: DebateState) -> str:
        if len(state["messages"]) > settings.message_threshold:
            return "summarize"
        if state["round"] >= state["max_rounds"]:
            return "judge"
        return "bear"

    return route


def make_route_after_summarize(settings: Settings) -> Callable[[DebateState], str]:
    def route(state: DebateState) -> str:
        if state["round"] >= state["max_rounds"]:
            return "judge"
        return "bear"

    return route
