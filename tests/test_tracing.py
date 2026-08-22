from bear_bull_debate.tracing import trace


async def test_trace_noop_returns_decorated_identity():
    decorated = trace("dummy")(_async_identity)

    # trace() must work even if langsmith is not installed/configured
    assert await decorated() == "ok"


async def _async_identity():
    return "ok"
