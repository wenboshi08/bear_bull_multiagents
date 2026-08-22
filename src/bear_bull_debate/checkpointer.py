from langgraph.checkpoint.memory import InMemorySaver

_MISSING_POSTGRES_MSG = (
    "PostgresSaver requires the 'postgres' extra. "
    "Install with: pip install 'bear-bull-debate[postgres]'"
)


async def make_checkpointer(uri: str | None):
    """Create a checkpointer: InMemorySaver (dev) or AsyncPostgresSaver (prod)."""
    if not uri:
        return InMemorySaver()
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:
        raise RuntimeError(_MISSING_POSTGRES_MSG) from exc
    saver = AsyncPostgresSaver.from_conn_string(uri)
    await saver.setup()
    return saver


async def close_checkpointer(checkpointer) -> None:
    """Close a checkpointer if it supports an async/sync close method."""
    close = getattr(checkpointer, "aclose", None) or getattr(
        checkpointer, "close", None
    )
    if close is None:
        return
    result = close()
    if hasattr(result, "__await__"):
        await result
