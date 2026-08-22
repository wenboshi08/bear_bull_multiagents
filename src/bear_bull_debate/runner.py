"""Convenience entry points for running a debate outside the FastAPI app.

Use these in plain scripts, REPLs, and Jupyter/Google-Colab notebooks where
standing up a full Uvicorn server is unnecessary.
"""

import asyncio

from .api import DebateRequest, initial_state
from .checkpointer import close_checkpointer, make_checkpointer
from .config import Settings
from .graph import build_graph


async def run_debate_async(
    company: str,
    max_rounds: int | None = None,
    settings: Settings | None = None,
) -> dict:
    """Run a full Bear-vs-Bull debate asynchronously.

    Args:
        company: Target company name (e.g. ``"AAPL"``).
        max_rounds: Number of debate rounds. ``None`` falls back to
            ``settings.max_rounds``. Must be in ``[1, 5]`` (validated by
            ``DebateRequest``).
        settings: Optional :class:`~bear_bull_debate.config.Settings`. Defaults
            to ``Settings.from_env()``.

    Returns:
        ``{"thread_id", "final_report", "tool_logs"}``.
    """
    settings = settings or Settings.from_env()
    if max_rounds is None:
        max_rounds = settings.max_rounds

    checkpointer = await make_checkpointer(settings.checkpointer_uri)
    try:
        graph = build_graph(settings, checkpointer=checkpointer)
        req = DebateRequest(company=company, max_rounds=max_rounds)
        config = {"configurable": {"thread_id": req.thread_id}}
        result = await graph.ainvoke(initial_state(req), config)
        messages = result["messages"]
        return {
            "thread_id": req.thread_id,
            "final_report": messages[-1].content,
            "tool_logs": result["tool_outputs"],
        }
    finally:
        await close_checkpointer(checkpointer)


def run_debate(
    company: str,
    max_rounds: int | None = None,
    settings: Settings | None = None,
) -> dict:
    """Synchronous wrapper around :func:`run_debate_async`.

    Works out-of-the-box in plain scripts. Inside a Jupyter/Colab notebook,
    apply ``nest_asyncio`` first (``import nest_asyncio; nest_asyncio.apply()``)
    so ``asyncio.run`` can be used within the kernel's already-running loop, or
    simply ``await run_debate_async(...)`` (IPython supports top-level await).
    """
    return asyncio.run(
        run_debate_async(company, max_rounds=max_rounds, settings=settings)
    )
