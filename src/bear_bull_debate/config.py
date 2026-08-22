import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    max_rounds: int = 2
    bear_model: str = "gpt-4o-mini"
    bull_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o"
    summary_model: str = "gpt-4o-mini"
    history_window: int = 4
    message_threshold: int = 12
    max_tool_rounds: int = 5
    checkpointer_uri: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            max_rounds=int(os.getenv("MAX_ROUNDS", "2")),
            bear_model=os.getenv("BEAR_MODEL", "gpt-4o-mini"),
            bull_model=os.getenv("BULL_MODEL", "gpt-4o-mini"),
            judge_model=os.getenv("JUDGE_MODEL", "gpt-4o"),
            summary_model=os.getenv("SUMMARY_MODEL", "gpt-4o-mini"),
            history_window=int(os.getenv("HISTORY_WINDOW", "4")),
            message_threshold=int(os.getenv("MESSAGE_THRESHOLD", "12")),
            max_tool_rounds=int(os.getenv("MAX_TOOL_ROUNDS", "5")),
            checkpointer_uri=os.getenv("DATABASE_URL") or None,
        )
