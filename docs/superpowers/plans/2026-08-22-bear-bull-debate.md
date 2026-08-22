# Bull-Bear Debate Stock Analysis System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph-based multi-round Bull vs Bear stock debate system with a Judge that emits a structured, bias-resistant investment report, backed by mock tools, a checkpointer, context-window management, and a FastAPI endpoint.

**Architecture:** A `StateGraph` runs `bear → bull → router` in a loop. The router sends long conversations through a `summarize` node (which physically frees old messages via `RemoveMessage`) and eventually to a `judge` node. `DebateState` uses `add_messages` for the message list and `operator.add` for `tool_outputs` so updates append rather than overwrite. A checkpointer (`InMemorySaver` for dev, `AsyncPostgresSaver` for prod) provides `thread_id` isolation and resume. The API layer validates input with Pydantic and drives `graph.ainvoke`.

**Tech Stack:** Python ≥3.11, `uv`, LangGraph, `langchain-openai`, `langchain-core`, `tenacity`, Pydantic v2, FastAPI, Uvicorn, pytest + pytest-asyncio, `FakeListChatModel` for deterministic tests.

**Spec:** The design document provided by the user ("多空辩论股票分析系统设计文档", v1.3) — this plan is derived from it and travels with it; executors should read both.

## Design-doc corrections applied (argue from the spec, but fix the drift)

The design doc is sound overall, but five details no longer match the current LangGraph/LangChain API surface, and one is an internal inconsistency. This plan implements the *intended* behavior with corrected imports/APIs:

1. **`ToolException` import** — the doc writes `from langchain_core.exceptions import ToolException`. The real location is `from langchain_core.tools import ToolException`.
2. **`MemorySaver` → `InMemorySaver`** — current LangGraph renamed it. Import: `from langgraph.checkpoint.memory import InMemorySaver`.
3. **`PostgresSaver` moved to its own package** — now `langgraph-checkpoint-postgres`; the async variant is `from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver` (used because the API is async). Kept as an optional extra so dev/testing needs no database.
4. **`max_rounds` is per-request in the doc (`DebateRequest`) but routing used a global `MAX_ROUNDS`.** Resolution: add `max_rounds` to `DebateState`, seed it from the request, and have routers read `state["max_rounds"]`. This keeps routing pure (depends only on state) and makes the per-request override actually work.
5. **Routers are factories, not module-level functions** — `make_route_after_bull(settings)` closes over `Settings.message_threshold`, avoiding global state and making routing trivially testable.

## Global Constraints

- Python **>= 3.11**; dependency management with **uv** (`uv sync`, `uv run`).
- `langgraph>=0.4.0`, `langchain-core>=0.3.0`, `langchain-openai>=0.2.0`, `tenacity>=8.2.0`, `pydantic>=2.7.0`, `fastapi>=0.111.0`, `uvicorn[standard]>=0.30.0`.
- Optional `postgres` extra: `langgraph-checkpoint-postgres>=2.0.0`, `psycopg[binary,pool]>=3.2.0`. Dev: `pytest`, `pytest-asyncio`, `httpx`, `ruff`.
- Default config (env-overridable, verbatim from spec §6): `MAX_ROUNDS=2`, `BEAR_MODEL=gpt-4o-mini`, `BULL_MODEL=gpt-4o-mini`, `JUDGE_MODEL=gpt-4o`, `HISTORY_WINDOW=4`, `MESSAGE_THRESHOLD=12`, `DATABASE_URL` empty → in-memory checkpointer.
- Company input validated against `^[a-zA-Z0-9\s\-\.]+$`, length 1–50; `max_rounds` in `[1,5]`.
- `tool_outputs` MUST be `Annotated[list[str], operator.add]`; `messages` MUST be `Annotated[list[BaseMessage], add_messages]`.
- Every `ToolMessage` MUST carry `tool_call_id` AND `name`.
- LLM calls go through a `tenacity` retry wrapper (3 attempts, exponential backoff) on transient OpenAI errors only.
- TDD: write the failing test, run it, implement, re-run, commit — for every task.

---

## File structure

```
bear_bull_debate/
├── pyproject.toml
├── .gitignore
├── .env.example
├── docker-compose.yml            # Postgres for PostgresSaver manual verification
├── README.md
├── src/bear_bull_debate/
│   ├── __init__.py
│   ├── config.py                 # Settings dataclass (env-driven)
│   ├── state.py                  # DebateState TypedDict + reducers
│   ├── prompts.py                # Bear/Bull/Judge/Summarize system prompts
│   ├── tools.py                  # 3 mock tools + TOOLS list
│   ├── llm.py                    # make_llm + ainvoke_with_retry (tenacity)
│   ├── nodes.py                  # researcher/summarize/judge node factories
│   ├── router.py                 # route_after_bull / route_after_summarize factories
│   ├── checkpointer.py           # make_checkpointer / close_checkpointer
│   ├── graph.py                  # build_graph (StateGraph assembly)
│   └── api.py                    # DebateRequest, run_debate, create_app
└── tests/
    ├── conftest.py
    ├── test_state.py
    ├── test_tools.py
    ├── test_llm.py
    ├── test_nodes.py
    ├── test_router.py
    ├── test_graph.py
    └── test_api.py
```

---

## Task 1: Project scaffolding + `Settings`

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `src/bear_bull_debate/__init__.py`, `src/bear_bull_debate/config.py`
- Test: `tests/conftest.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `Settings` dataclass with fields `max_rounds, bear_model, bull_model, judge_model, summary_model, history_window, message_threshold, checkpointer_uri` and classmethod `Settings.from_env()`. Every later task imports it via `from bear_bull_debate.config import Settings`.
- Produces: the `settings` pytest fixture (returns a `Settings` with `checkpointer_uri=None`) used by all later tests.

- [ ] **Step 1: Initialize the project**

```bash
cd /Users/wenboshi/ml/bear_bull_debate
git init -b main
mkdir -p src/bear_bull_debate tests
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "bear-bull-debate"
version = "0.1.0"
description = "LLM-powered Bull/Bear debate stock analysis system built on LangGraph"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.4.0",
    "langchain-core>=0.3.0",
    "langchain-openai>=0.2.0",
    "tenacity>=8.2.0",
    "pydantic>=2.7.0",
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.30.0",
]

[project.optional-dependencies]
postgres = [
    "langgraph-checkpoint-postgres>=2.0.0",
    "psycopg[binary,pool]>=3.2.0",
]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.27.0",
    "ruff>=0.5.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/bear_bull_debate"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
pythonpath = ["src"]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 3: Write `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
.env
.pytest_cache/
.ruff_cache/
*.egg-info/
dist/
build/
```

- [ ] **Step 4: Write `.env.example`**

```bash
# LLM
OPENAI_API_KEY=sk-...
# LangSmith (optional)
LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=false
LANGCHAIN_PROJECT=bear-bull-debate
# Model routing
BEAR_MODEL=gpt-4o-mini
BULL_MODEL=gpt-4o-mini
JUDGE_MODEL=gpt-4o
SUMMARY_MODEL=gpt-4o-mini
# Debate controls
MAX_ROUNDS=2
HISTORY_WINDOW=4
MESSAGE_THRESHOLD=12
# Checkpointer (leave empty for in-memory)
DATABASE_URL=
```

- [ ] **Step 5: Write `src/bear_bull_debate/__init__.py`**

```python
"""Bull-Bear debate stock analysis system."""
```

- [ ] **Step 6: Write `src/bear_bull_debate/config.py`**

```python
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
            checkpointer_uri=os.getenv("DATABASE_URL") or None,
        )
```

- [ ] **Step 7: Write the failing test `tests/test_config.py`**

```python
from bear_bull_debate.config import Settings


def test_settings_defaults():
    s = Settings()
    assert s.max_rounds == 2
    assert s.judge_model == "gpt-4o"
    assert s.history_window == 4
    assert s.message_threshold == 12
    assert s.checkpointer_uri is None


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("MAX_ROUNDS", "4")
    monkeypatch.setenv("JUDGE_MODEL", "gpt-4o-2024-08-06")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/db")
    s = Settings.from_env()
    assert s.max_rounds == 4
    assert s.judge_model == "gpt-4o-2024-08-06"
    assert s.checkpointer_uri == "postgresql://localhost/db"
```

- [ ] **Step 8: Write `tests/conftest.py`**

```python
import pytest

from bear_bull_debate.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(checkpointer_uri=None)
```

- [ ] **Step 9: Install deps and run tests**

```bash
uv sync --extra dev
uv run pytest -q
```

Expected: 2 passed.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "chore: scaffold project and add Settings config"
```

---

## Task 2: `DebateState` and reducers

**Files:**
- Create: `src/bear_bull_debate/state.py`
- Test: `tests/test_state.py`

**Interfaces:**
- Produces: `DebateState` (a `TypedDict`) with fields `messages: Annotated[list[BaseMessage], add_messages]`, `round: int`, `company: str`, `summary: str`, `tool_outputs: Annotated[list[str], operator.add]`, `max_rounds: int`. All nodes and the graph use this as their state schema.
- Consumes: `Settings` (Task 1) — not needed here, but the `settings` fixture is reused in later tasks.

- [ ] **Step 1: Write `src/bear_bull_debate/state.py`**

```python
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
```

- [ ] **Step 2: Write the failing test `tests/test_state.py`**

```python
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
```

- [ ] **Step 3: Run tests to verify**

```bash
uv run pytest tests/test_state.py -q
```

Expected: all pass (the reducer behaviors are library-level; this locks the contract that later nodes rely on).

- [ ] **Step 4: Commit**

```bash
git add src/bear_bull_debate/state.py tests/test_state.py
git commit -m "feat: define DebateState with append reducers"
```

---

## Task 3: Mock tools

**Files:**
- Create: `src/bear_bull_debate/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Produces: `TOOLS` (a `list` of three `@tool`-decorated functions) and the tools `get_stock_price`, `get_financials`, `get_news_sentiment`, each taking `company: str` and returning a `str`; they raise `ToolException` on an empty `company`. The researcher node (Task 5) builds `{t.name: t for t in TOOLS}` and calls `tool.ainvoke(tc["args"])`.

- [ ] **Step 1: Write the failing test `tests/test_tools.py`**

```python
import pytest
from langchain_core.tools import ToolException

from bear_bull_debate.tools import get_financials, get_stock_price, get_news_sentiment


def test_get_stock_price_returns_mock_value():
    result = get_stock_price.invoke({"company": "AAPL"})
    assert "AAPL" in result
    assert "mock" in result


def test_get_financials_returns_mock_value():
    result = get_financials.invoke({"company": "TSLA"})
    assert "TSLA" in result
    assert "EPS" in result


def test_get_news_sentiment_returns_mock_value():
    result = get_news_sentiment.invoke({"company": "NVDA"})
    assert "NVDA" in result


def test_tool_raises_tool_exception_on_empty_company():
    with pytest.raises(ToolException):
        get_stock_price.invoke({"company": ""})
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_tools.py -q
```

Expected: FAIL (module `bear_bull_debate.tools` not found).

- [ ] **Step 3: Write `src/bear_bull_debate/tools.py`**

```python
from langchain_core.tools import ToolException, tool


@tool
def get_stock_price(company: str) -> str:
    """Get the latest mock stock price for a company."""
    company = (company or "").strip()
    if not company:
        raise ToolException("company parameter must not be empty")
    return f"{company} latest price: $123.45 (mock)"


@tool
def get_financials(company: str) -> str:
    """Get mock financial metrics (revenue, EPS, YoY growth) for a company."""
    company = (company or "").strip()
    if not company:
        raise ToolException("company parameter must not be empty")
    return f"{company} FY revenue: $10.0B, EPS: $3.20, YoY growth: +8% (mock)"


@tool
def get_news_sentiment(company: str) -> str:
    """Get mock recent news sentiment for a company."""
    company = (company or "").strip()
    if not company:
        raise ToolException("company parameter must not be empty")
    return f"{company} recent news sentiment: slightly negative (mock)"


TOOLS = [get_stock_price, get_financials, get_news_sentiment]
```

- [ ] **Step 4: Run to verify it passes**

```bash
uv run pytest tests/test_tools.py -q
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bear_bull_debate/tools.py tests/test_tools.py
git commit -m "feat: add mock stock-analysis tools"
```

---

## Task 4: LLM factory + retry wrapper

**Files:**
- Create: `src/bear_bull_debate/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces: `make_llm(model: str, temperature: float = 0.0) -> ChatOpenAI` (used by `graph.py`), and `ainvoke_with_retry(llm, messages: list[BaseMessage]) -> BaseMessage` (an async function wrapped in `tenacity`, used by every node).
- Consumes: `tenacity`, `openai` (transitive dep of langchain-openai) for the retryable exception types.

- [ ] **Step 1: Write the failing test `tests/test_llm.py`**

```python
import httpx
import pytest
from openai import APIConnectionError

from bear_bull_debate.llm import ainvoke_with_retry


class FlakyLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        raise APIConnectionError(
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        )


class ValueErrLLM:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        raise ValueError("boom")


async def test_retries_transient_errors_three_times():
    llm = FlakyLLM()
    with pytest.raises(APIConnectionError):
        await ainvoke_with_retry(llm, [])
    assert llm.calls == 3


async def test_does_not_retry_non_transient_errors():
    llm = ValueErrLLM()
    with pytest.raises(ValueError):
        await ainvoke_with_retry(llm, [])
    assert llm.calls == 1
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_llm.py -q
```

Expected: FAIL (`bear_bull_debate.llm` not found).

- [ ] **Step 3: Write `src/bear_bull_debate/llm.py`**

```python
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

RETRYABLE_EXCEPTIONS = (APIConnectionError, APITimeoutError, RateLimitError)


def make_llm(model: str, temperature: float = 0.0) -> ChatOpenAI:
    """Create a ChatOpenAI instance for the given model name."""
    return ChatOpenAI(model=model, temperature=temperature)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
)
async def ainvoke_with_retry(
    llm: BaseChatModel, messages: list[BaseMessage]
) -> BaseMessage:
    """Invoke the LLM, retrying transient network/rate-limit errors up to 3 times."""
    return await llm.ainvoke(messages)
```

- [ ] **Step 4: Run to verify it passes**

```bash
uv run pytest tests/test_llm.py -q
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bear_bull_debate/llm.py tests/test_llm.py
git commit -m "feat: add LLM factory and tenacity retry wrapper"
```

---

## Task 5: Researcher node (Bear/Bull)

**Files:**
- Create: `src/bear_bull_debate/prompts.py`, `src/bear_bull_debate/nodes.py`
- Test: `tests/test_nodes.py` (researcher tests only; summarize/judge tests added in Tasks 6–7)

**Interfaces:**
- Produces: `make_researcher_node(role: str, llm, tools, settings) -> Callable` returning an `async def node(state: DebateState) -> dict`. For `role == "bear"` it returns `{"messages": [...], "tool_outputs": [...]}`; for `"bull"` it also returns `{"round": state["round"] + 1}`.
- Consumes: `Settings` (Task 1), `DebateState` (Task 2), `TOOLS` (Task 3), `ainvoke_with_retry` (Task 4), prompts from `prompts.py`.

- [ ] **Step 1: Write `src/bear_bull_debate/prompts.py`**

```python
BEAR_SYSTEM_PROMPT = """\
You are the BEAR researcher in a structured investment debate about {company}.
Your mission is to argue the bearish (negative) case as forcefully and honestly as you can.
Rules:
- Ground every claim in data. Use the provided tools (get_stock_price, get_financials, get_news_sentiment) to fetch evidence before making a claim that depends on data.
- Respond to the Bull's arguments where possible, but stay focused on the bear case.
- Be concise and specific. End with a clear bearish thesis.
"""

BULL_SYSTEM_PROMPT = """\
You are the BULL researcher in a structured investment debate about {company}.
Your mission is to argue the bullish (positive) case as forcefully and honestly as you can.
Rules:
- Ground every claim in data. Use the provided tools (get_stock_price, get_financials, get_news_sentiment) to fetch evidence before making a claim that depends on data.
- Respond to the Bear's arguments where possible, but stay focused on the bull case.
- Be concise and specific. End with a clear bullish thesis.
"""

SUMMARY_SYSTEM_PROMPT = """\
You condense an ongoing investment debate into a dense, lossless summary.
Preserve every factual claim, data point, tool result, and the core argument of BOTH the Bear and the Bull.
Do not take sides. Keep the summary compact.
"""

JUDGE_SYSTEM_PROMPT = """\
You are the impartial judge of a debate between a Bear and a Bull researcher about {company}.
Your job is to produce a final, balanced investment recommendation.

To counter recency and length bias, follow this process strictly:
1. List the Bear's strongest arguments and the evidence behind each.
2. List the Bull's strongest arguments and the evidence behind each.
3. Compare them point by point, noting which side has better evidence on each contested point.
4. State which side has the stronger overall case and why.

Output a structured report in this exact markdown format:

## Verdict
<one sentence: Bullish / Bearish / Neutral on {company}>

## Bear's Case
<summary of strongest bear arguments>

## Bull's Case
<summary of strongest bull arguments>

## Head-to-Head
<point-by-point comparison>

## Recommendation
<actionable recommendation with position and reasoning>

## Confidence
<Low / Medium / High, with one-line justification>
"""
```

- [ ] **Step 2: Write the failing test (researcher section) in `tests/test_nodes.py`**

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from bear_bull_debate.nodes import make_researcher_node
from bear_bull_debate.tools import TOOLS


def make_state(**overrides):
    state = {
        "messages": [HumanMessage(content="Debate topic: AAPL", id="seed")],
        "round": 0,
        "company": "AAPL",
        "summary": "",
        "tool_outputs": [],
        "max_rounds": 2,
    }
    state.update(overrides)
    return state


async def test_researcher_no_tool_calls(settings):
    llm = FakeListChatModel(
        responses=[AIMessage(content="Bear argument: AAPL overvalued")]
    )
    node = make_researcher_node("bear", llm, TOOLS, settings)
    result = await node(make_state())
    assert result["messages"][-1].content == "Bear argument: AAPL overvalued"
    assert "round" not in result


async def test_researcher_executes_tool_and_logs_output(settings):
    llm = FakeListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_stock_price",
                        "args": {"company": "AAPL"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="After checking price: bear thesis"),
        ]
    )
    node = make_researcher_node("bear", llm, TOOLS, settings)
    result = await node(make_state())

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call_1"
    assert "123.45" in tool_messages[0].content
    assert result["tool_outputs"] == [
        "[get_stock_price] AAPL latest price: $123.45 (mock)"
    ]


async def test_researcher_tool_exception_degrades_gracefully(settings):
    llm = FakeListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_stock_price",
                        "args": {"company": ""},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Tool failed, arguing from general knowledge."),
        ]
    )
    node = make_researcher_node("bear", llm, TOOLS, settings)
    result = await node(make_state())

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Tool error" in tool_messages[0].content
    assert (
        result["messages"][-1].content
        == "Tool failed, arguing from general knowledge."
    )


async def test_bull_increments_round(settings):
    llm = FakeListChatModel(responses=[AIMessage(content="Bull argument")])
    node = make_researcher_node("bull", llm, TOOLS, settings)
    result = await node(make_state(round=1))
    assert result["round"] == 2
```

- [ ] **Step 3: Run to verify it fails**

```bash
uv run pytest tests/test_nodes.py -q
```

Expected: FAIL (`bear_bull_debate.nodes` not found).

- [ ] **Step 4: Write `src/bear_bull_debate/nodes.py`**

```python
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

    return node
```

- [ ] **Step 5: Run to verify it passes**

```bash
uv run pytest tests/test_nodes.py -q
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/bear_bull_debate/prompts.py src/bear_bull_debate/nodes.py tests/test_nodes.py
git commit -m "feat: add Bear/Bull researcher node with tool-calling loop"
```

---

## Task 6: Summarize node (context compaction via `RemoveMessage`)

**Files:**
- Modify: `src/bear_bull_debate/nodes.py` (append `make_summarize_node`)
- Test: `tests/test_nodes.py` (append summarize tests)

**Interfaces:**
- Produces: `make_summarize_node(summary_llm, settings) -> Callable` returning `async def node(state) -> dict` that returns `{"summary": <str>, "messages": [RemoveMessage, ...]}`. When `len(messages) <= settings.history_window` it is a no-op returning `{"summary": state["summary"]}`.
- Consumes: `ainvoke_with_retry` (Task 4), `SUMMARY_SYSTEM_PROMPT` (Task 5), `RemoveMessage`.

- [ ] **Step 1: Append the failing tests to `tests/test_nodes.py`**

```python
from bear_bull_debate.nodes import make_summarize_node  # add to top imports


async def test_summarize_removes_old_messages(settings):
    old = [
        HumanMessage(content="Debate topic: AAPL", id="s1"),
        AIMessage(content="Bear R1", id="m1"),
        AIMessage(content="Bull R1", id="m2"),
        AIMessage(content="Bear R2", id="m3"),
        AIMessage(content="Bull R2", id="m4"),
    ]
    llm = FakeListChatModel(responses=[AIMessage(content="Compressed summary")])
    node = make_summarize_node(llm, settings)
    result = await node(make_state(messages=old, summary="prior"))

    assert result["summary"] == "Compressed summary"
    removed_ids = {m.id for m in result["messages"] if isinstance(m, RemoveMessage)}
    # history_window=4 keeps the last 4 (m1..m4); only the seed "s1" is removed
    assert removed_ids == {"s1"}


async def test_summarize_noop_when_short(settings):
    short = [HumanMessage(content="hi", id="s1"), AIMessage(content="a", id="m1")]
    llm = FakeListChatModel(responses=[AIMessage(content="should not be called")])
    node = make_summarize_node(llm, settings)
    result = await node(make_state(messages=short, summary="prior"))
    assert result["summary"] == "prior"
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_nodes.py -q
```

Expected: FAIL (`make_summarize_node` not defined).

- [ ] **Step 3: Append `make_summarize_node` to `src/bear_bull_debate/nodes.py`**

```python
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

    return node
```

- [ ] **Step 4: Run to verify it passes**

```bash
uv run pytest tests/test_nodes.py -q
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bear_bull_debate/nodes.py tests/test_nodes.py
git commit -m "feat: add summarize node with RemoveMessage compaction"
```

---

## Task 7: Judge node

**Files:**
- Modify: `src/bear_bull_debate/nodes.py` (append `make_judge_node`)
- Test: `tests/test_nodes.py` (append judge test)

**Interfaces:**
- Produces: `make_judge_node(judge_llm) -> Callable` returning `async def node(state) -> dict` that appends the final report as `{"messages": [AIMessage]}`. The report is the last message in state, which the API reads as `final_report`.
- Consumes: `ainvoke_with_retry` (Task 4), `JUDGE_SYSTEM_PROMPT` (Task 5), `_format_messages` (Task 6).

- [ ] **Step 1: Append the failing test to `tests/test_nodes.py`**

```python
from bear_bull_debate.nodes import make_judge_node  # add to top imports


async def test_judge_appends_report(settings):
    llm = FakeListChatModel(responses=[AIMessage(content="## Verdict\nNeutral")])
    node = make_judge_node(llm)
    result = await node(make_state())
    assert result["messages"][-1].content.startswith("## Verdict")
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_nodes.py -q
```

Expected: FAIL (`make_judge_node` not defined).

- [ ] **Step 3: Append `make_judge_node` to `src/bear_bull_debate/nodes.py`**

```python
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

    return node
```

- [ ] **Step 4: Run to verify it passes**

```bash
uv run pytest tests/test_nodes.py -q
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bear_bull_debate/nodes.py tests/test_nodes.py
git commit -m "feat: add judge node producing the final report"
```

---

## Task 8: Router (conditional edges)

**Files:**
- Create: `src/bear_bull_debate/router.py`
- Test: `tests/test_router.py`

**Interfaces:**
- Produces: `make_route_after_bull(settings) -> Callable[[DebateState], str]` returning one of `"summarize" | "judge" | "bear"`, and `make_route_after_summarize(settings) -> Callable[[DebateState], str]` returning `"judge" | "bear"`.
- Consumes: `Settings.message_threshold` (Task 1), `DebateState` (Task 2).

- [ ] **Step 1: Write the failing test `tests/test_router.py`**

```python
from bear_bull_debate.router import make_route_after_bull, make_route_after_summarize


def make_state(messages_len, round_, max_rounds=2):
    return {
        "messages": list(range(messages_len)),
        "round": round_,
        "max_rounds": max_rounds,
    }


def test_route_after_bull_summarize(settings):
    route = make_route_after_bull(settings)
    assert route(make_state(13, 0)) == "summarize"


def test_route_after_bull_judge(settings):
    route = make_route_after_bull(settings)
    assert route(make_state(5, 2)) == "judge"


def test_route_after_bull_continue(settings):
    route = make_route_after_bull(settings)
    assert route(make_state(5, 1)) == "bear"


def test_route_after_summarize_judge(settings):
    route = make_route_after_summarize(settings)
    assert route(make_state(0, 2)) == "judge"


def test_route_after_summarize_continue(settings):
    route = make_route_after_summarize(settings)
    assert route(make_state(0, 1)) == "bear"
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_router.py -q
```

Expected: FAIL (`bear_bull_debate.router` not found).

- [ ] **Step 3: Write `src/bear_bull_debate/router.py`**

```python
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
```

- [ ] **Step 4: Run to verify it passes**

```bash
uv run pytest tests/test_router.py -q
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bear_bull_debate/router.py tests/test_router.py
git commit -m "feat: add debate routing logic"
```

---

## Task 9: Graph assembly + checkpointer

**Files:**
- Create: `src/bear_bull_debate/checkpointer.py`, `src/bear_bull_debate/graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Produces: `build_graph(settings, checkpointer=None, models=None, interrupt_before=None) -> CompiledStateGraph` with nodes `bear`, `bull`, `summarize`, `judge`; `models` is an optional `dict` keyed by `"bear" | "bull" | "judge" | "summary"` for test injection (skips real `make_llm`). Also `make_checkpointer(uri) -> checkpointer` and `close_checkpointer(checkpointer)`.
- Consumes: everything from Tasks 1–8.

- [ ] **Step 1: Write the failing test `tests/test_graph.py`**

```python
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from bear_bull_debate.checkpointer import make_checkpointer
from bear_bull_debate.config import Settings
from bear_bull_debate.graph import build_graph


def make_initial(company="AAPL", max_rounds=2):
    return {
        "company": company,
        "round": 0,
        "max_rounds": max_rounds,
        "messages": [HumanMessage(content=f"Debate topic: {company}", id="seed")],
        "summary": "",
        "tool_outputs": [],
    }


def make_models(rounds=2):
    return {
        "bear": FakeListChatModel(
            responses=[AIMessage(content=f"Bear R{i}") for i in range(1, rounds + 1)]
        ),
        "bull": FakeListChatModel(
            responses=[AIMessage(content=f"Bull R{i}") for i in range(1, rounds + 1)]
        ),
        "judge": FakeListChatModel(responses=[AIMessage(content="FINAL REPORT")]),
        "summary": FakeListChatModel(responses=[AIMessage(content="SUMMARY")]),
    }


async def test_graph_runs_two_rounds_then_judge(settings):
    graph = build_graph(settings, checkpointer=None, models=make_models(rounds=2))
    result = await graph.ainvoke(
        make_initial(), config={"configurable": {"thread_id": "t1"}}
    )

    assert result["round"] == 2
    assert result["messages"][-1].content == "FINAL REPORT"
    assert result["tool_outputs"] == []


async def test_graph_summarizes_when_threshold_exceeded():
    settings = Settings(
        max_rounds=3, history_window=2, message_threshold=4, checkpointer_uri=None
    )
    graph = build_graph(settings, checkpointer=None, models=make_models(rounds=3))
    result = await graph.ainvoke(
        make_initial(max_rounds=3), config={"configurable": {"thread_id": "t2"}}
    )

    assert result["summary"] == "SUMMARY"
    assert result["round"] == 3
    assert result["messages"][-1].content == "FINAL REPORT"
    # summarize compacted the early messages
    assert len(result["messages"]) < 7


async def test_resume_from_interrupt_before_judge(settings):
    from langgraph.checkpoint.memory import InMemorySaver

    graph = build_graph(
        settings,
        checkpointer=InMemorySaver(),
        models=make_models(rounds=2),
        interrupt_before=["judge"],
    )
    config = {"configurable": {"thread_id": "thread-2"}}
    first = await graph.ainvoke(make_initial(), config)
    assert first["round"] == 2

    final = await graph.ainvoke(None, config)
    assert final["messages"][-1].content == "FINAL REPORT"


async def test_checkpointer_factory_memory():
    cp = await make_checkpointer(None)
    assert cp is not None


async def test_checkpointer_factory_postgres_missing_extra(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres.aio", None)
    with pytest.raises(RuntimeError, match="postgres"):
        await make_checkpointer("postgresql://localhost/db")
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_graph.py -q
```

Expected: FAIL (`bear_bull_debate.graph` not found).

- [ ] **Step 3: Write `src/bear_bull_debate/checkpointer.py`**

```python
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
```

- [ ] **Step 4: Write `src/bear_bull_debate/graph.py`**

```python
from typing import Any

from langgraph.graph import END, START, StateGraph

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
    bear_llm = models.get("bear") or make_llm(settings.bear_model).bind_tools(TOOLS)
    bull_llm = models.get("bull") or make_llm(settings.bull_model).bind_tools(TOOLS)
    judge_llm = models.get("judge") or make_llm(settings.judge_model)
    summary_llm = models.get("summary") or make_llm(settings.summary_model)

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
```

- [ ] **Step 5: Run to verify it passes**

```bash
uv run pytest tests/test_graph.py -q
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/bear_bull_debate/checkpointer.py src/bear_bull_debate/graph.py tests/test_graph.py
git commit -m "feat: assemble debate graph with checkpointer support"
```

---

## Task 10: FastAPI endpoint + input validation

**Files:**
- Create: `src/bear_bull_debate/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: `DebateRequest` (Pydantic model: `company`, `thread_id`, `max_rounds`), `initial_state(req) -> dict`, `run_debate(graph, req) -> dict` returning `{"thread_id", "final_report", "tool_logs"}`, and `create_app(settings=None, graph=None) -> FastAPI` with `POST /api/v1/debate` and `GET /healthz`.
- Consumes: `build_graph` (Task 9), `make_checkpointer` / `close_checkpointer` (Task 9), `Settings` (Task 1).

- [ ] **Step 1: Write the failing test `tests/test_api.py`**

```python
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from bear_bull_debate.api import DebateRequest, create_app, run_debate


class StubGraph:
    def __init__(self, result):
        self.result = result

    async def ainvoke(self, state, config):
        self.last_state = state
        self.last_config = config
        return self.result


def stub_graph():
    return StubGraph(
        {"messages": [AIMessage(content="FINAL REPORT")], "tool_outputs": []}
    )


async def test_run_debate_returns_final_report():
    graph = stub_graph()
    req = DebateRequest(company="AAPL", thread_id="t-1", max_rounds=2)
    out = await run_debate(graph, req)
    assert out["final_report"] == "FINAL REPORT"
    assert out["thread_id"] == "t-1"
    assert out["tool_logs"] == []


def test_debate_request_rejects_invalid_company():
    with pytest.raises(ValidationError):
        DebateRequest(company="BAD; rm -rf /")
    with pytest.raises(ValidationError):
        DebateRequest(company="")


def test_debate_request_rejects_out_of_range_rounds():
    with pytest.raises(ValidationError):
        DebateRequest(company="AAPL", max_rounds=9)


def test_debate_request_accepts_valid_input():
    req = DebateRequest(company="AAPL", max_rounds=3)
    assert req.max_rounds == 3
    assert req.thread_id


def test_api_endpoint_smoke():
    app = create_app(graph=stub_graph())
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/debate", json={"company": "AAPL", "max_rounds": 2}
        )
    assert resp.status_code == 200
    assert resp.json()["final_report"] == "FINAL REPORT"


def test_api_endpoint_rejects_invalid_company():
    app = create_app(graph=stub_graph())
    with TestClient(app) as client:
        resp = client.post("/api/v1/debate", json={"company": "BAD; DROP TABLE"})
    assert resp.status_code == 422


def test_healthz():
    app = create_app(graph=stub_graph())
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_api.py -q
```

Expected: FAIL (`bear_bull_debate.api` not found).

- [ ] **Step 3: Write `src/bear_bull_debate/api.py`**

```python
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from .checkpointer import close_checkpointer, make_checkpointer
from .config import Settings
from .graph import build_graph


class DebateRequest(BaseModel):
    company: str = Field(
        ...,
        min_length=1,
        max_length=50,
        pattern=r"^[a-zA-Z0-9\s\-\.]+$",
    )
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    max_rounds: int = Field(default=2, ge=1, le=5)


def initial_state(req: DebateRequest) -> dict:
    return {
        "company": req.company,
        "round": 0,
        "max_rounds": req.max_rounds,
        "messages": [
            HumanMessage(
                content=(
                    f"Debate topic: {req.company}. "
                    "Bear opens, Bull rebuts. Fetch data with tools when useful."
                ),
                id="debate-seed",
            )
        ],
        "summary": "",
        "tool_outputs": [],
    }


async def run_debate(graph: Any, req: DebateRequest) -> dict:
    config = {"configurable": {"thread_id": req.thread_id}}
    result = await graph.ainvoke(initial_state(req), config)
    messages = result["messages"]
    return {
        "thread_id": req.thread_id,
        "final_report": messages[-1].content,
        "tool_logs": result["tool_outputs"],
    }


def create_app(settings: Settings | None = None, graph: Any = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if graph is not None:
            app.state.graph = graph
        else:
            checkpointer = await make_checkpointer(settings.checkpointer_uri)
            app.state.graph = build_graph(settings, checkpointer=checkpointer)
            app.state.checkpointer = checkpointer
        yield
        if graph is None:
            await close_checkpointer(app.state.checkpointer)

    app = FastAPI(title="Bull-Bear Debate API", lifespan=lifespan)

    @app.post("/api/v1/debate")
    async def start_debate(req: DebateRequest) -> dict:
        return await run_debate(app.state.graph, req)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
```

- [ ] **Step 4: Run to verify it passes**

```bash
uv run pytest tests/test_api.py -q
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/bear_bull_debate/api.py tests/test_api.py
git commit -m "feat: add FastAPI endpoint with Pydantic input validation"
```

---

## Task 11: Observability, docs, and Postgres verification

**Files:**
- Create: `src/bear_bull_debate/tracing.py`, `docker-compose.yml`, `README.md`
- Modify: `src/bear_bull_debate/nodes.py` (decorate node functions), `src/bear_bull_debate/llm.py` (optional console callback)
- Test: `tests/test_tracing.py`

**Interfaces:**
- Produces: `trace(name: str)` — a safe wrapper around `langsmith.traceable` that is a no-op when `langsmith` is unavailable; applied to the inner async `node` functions in `nodes.py`.
- Consumes: everything prior. This is the final integration/polish task.

- [ ] **Step 1: Write the failing test `tests/test_tracing.py`**

```python
from bear_bull_debate.tracing import trace


async def test_trace_noop_returns_decorated_identity():
    decorated = trace("dummy")(_async_identity)

    # trace() must work even if langsmith is not installed/configured
    assert await decorated() == "ok"


async def _async_identity():
    return "ok"
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest tests/test_tracing.py -q
```

Expected: FAIL (`bear_bull_debate.tracing` not found).

- [ ] **Step 3: Write `src/bear_bull_debate/tracing.py`**

```python
try:
    from langsmith import traceable as _langsmith_traceable
except ImportError:  # pragma: no cover - exercised when langsmith is absent
    _langsmith_traceable = None


def trace(name: str):
    """Return a decorator that enables LangSmith tracing when available."""
    if _langsmith_traceable is None:
        return lambda fn: fn
    return _langsmith_traceable(name=name)
```

- [ ] **Step 4: Decorate the node functions in `src/bear_bull_debate/nodes.py`**

Add the import and wrap each node before returning it:

```python
from .tracing import trace  # add near the other local imports
```

In `make_researcher_node`, just before `return node`:

```python
    node = trace(f"{role}_researcher")(node)
    return node
```

In `make_summarize_node`, just before `return node`:

```python
    node = trace("summarize")(node)
    return node
```

In `make_judge_node`, just before `return node`:

```python
    node = trace("judge")(node)
    return node
```

- [ ] **Step 5: Add optional console callback to `src/bear_bull_debate/llm.py`**

```python
import os  # add at top

def make_llm(model: str, temperature: float = 0.0) -> ChatOpenAI:
    """Create a ChatOpenAI instance for the given model name."""
    kwargs = {"model": model, "temperature": temperature}
    if os.getenv("VERBOSE") == "1":
        from langchain_core.callbacks import ConsoleCallbackHandler

        kwargs["callbacks"] = [ConsoleCallbackHandler()]
    return ChatOpenAI(**kwargs)
```

- [ ] **Step 6: Write `docker-compose.yml`**

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: debate
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

- [ ] **Step 7: Write `README.md`**

```markdown
# Bull-Bear Debate Stock Analysis System

A LangGraph-based multi-round Bull vs Bear debate with a Judge that emits a structured,
bias-resistant investment report. Tools are mocked in V1; the debate mechanism and agent
coordination are the focus.

## Quickstart

```bash
uv sync --extra dev
cp .env.example .env   # set OPENAI_API_KEY
uv run uvicorn bear_bull_debate.api:app --reload
curl -X POST http://127.0.0.1:8000/api/v1/debate \
  -H 'Content-Type: application/json' \
  -d '{"company": "AAPL", "max_rounds": 2}'
```

## Test

```bash
uv run pytest -q
```

## Production checkpointer (Postgres)

```bash
uv sync --extra dev --extra postgres
docker compose up -d postgres
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/debate
uv run uvicorn bear_bull_debate.api:app --reload
```

The app lazily imports `AsyncPostgresSaver` only when `DATABASE_URL` is set and runs
`checkpointer.setup()` on startup to create tables.

## Observability

Set `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` for LangSmith traces (each node is
decorated with `@traceable`). Set `VERBOSE=1` for console callback output.
```

- [ ] **Step 8: Run the full suite**

```bash
uv run pytest -q
```

Expected: all tests pass (26 total).

- [ ] **Step 9: Manual Postgres verification (documented, not automated)**

```bash
uv sync --extra dev --extra postgres
docker compose up -d postgres
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/debate \
  uv run uvicorn bear_bull_debate.api:app --reload
```

Confirm startup logs "Creating tables" / no import error, then `curl` the endpoint twice with the same `thread_id` and confirm the second call resumes the checkpoint.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "chore: add observability, docs, and Postgres verification"
```

---

## Self-review notes

- **Spec coverage:** §3.1 (state/reducers) → Task 2; §3.2 nodes → Tasks 5–7; §3.3 routing → Task 8; §3.4 retry/observability/validation → Tasks 4, 10, 11; §4 data flow → Task 9 graph test; §5 API → Task 10; §6 config → Task 1; §8 test strategy (reducer, RemoveMessage, ToolException, routing, persistence/resume) → Tasks 2, 6, 5, 8, 9. §7.2 HITL and §11 future work (Critic/Reflection, multimodal, RLHF) are intentionally out of V1 scope; `interrupt_before` resume (Task 9) lays the checkpoint foundation for HITL.
- **Placeholder scan:** no TBD/TODO; every code step contains full content.
- **Type consistency:** `DebateState` keys (`messages`, `round`, `company`, `summary`, `tool_outputs`, `max_rounds`) are identical across `state.py`, `nodes.py`, `router.py`, `graph.py`, and all tests. `make_researcher_node(role, llm, tools, settings)`, `make_summarize_node(llm, settings)`, `make_judge_node(llm)`, `build_graph(settings, checkpointer, models, interrupt_before)`, `make_checkpointer(uri)`, `run_debate(graph, req)`, `create_app(settings, graph)` signatures are stable across producer and consumer tasks.
