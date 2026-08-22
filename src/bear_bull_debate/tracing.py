try:
    from langsmith import traceable as _langsmith_traceable
except ImportError:  # pragma: no cover - exercised when langsmith is absent
    _langsmith_traceable = None


def trace(name: str):
    """Return a decorator that enables LangSmith tracing when available."""
    if _langsmith_traceable is None:
        return lambda fn: fn
    return _langsmith_traceable(name=name)
