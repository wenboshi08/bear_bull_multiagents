# Bear-Bull Debate Stock Analysis System — System Design

**Version:** 1.0
**Date:** 2026-08-22
**Status:** Implemented

---

## 1. Overview

Traditional stock analysis relies on a single analyst's viewpoint, which invites
confirmation bias. This system instead simulates an institutional debate: a **Bear
researcher** and a **Bull researcher** argue multiple rounds, then an impartial
**Judge** weighs both sides and emits a structured, bias-resistant recommendation.

The system is built on the **LangChain** ecosystem for the model/message/tool layer
and **LangGraph** for the orchestration layer (stateful graph, reducers, checkpointing,
and interrupt/resume). The current scope focuses on the *debate mechanism and agent
coordination pattern*; data access is stubbed with mock tools.

### 1.1 Goals

- Multi-round Bear vs Bull debate with a Judge that produces a structured report.
- Official LangGraph best practices: `add_messages`, `RemoveMessage`, checkpointer,
  `interrupt_before` for breakpoints.
- A robust tool-calling loop with graceful degradation and a hard iteration cap.
- Engineering safeguards: context-window management, retry, input validation, tracing.

### 1.2 Non-goals (V1)

- Real market data (V2 replaces mock tools with `yfinance` / `Tavily` / `Alpha Vantage`).
- Full human-in-the-loop (the checkpoint/interrupt foundation is in place; a UI is not).
- Reflection/critique of the Judge's report (future work).

---

## 2. Architecture

The system is a **LangGraph `StateGraph`**. A graph is a directed graph of *nodes*
(computation steps) connected by *edges*; LangGraph executes it, maintaining a
*state* object that flows through the graph and is persisted by a *checkpointer*.

```
              ┌────────────────────────────────────────────────────────────┐
              │                      StateGraph (DebateState)              │
              │                                                            │
  START ──▶ [bear] ──▶ [bull] ──▶ (router) ──▶ END                          │
               ▲                    │  │                                    │
               │                    │  ├── len(messages) > threshold ──▶ [summarize] ─┐
               │                    │  │                                          │
               └────────────────────┘  └── round >= max_rounds ──▶ [judge] ◀───────┘
                    (continue)
```

- **Nodes:** `bear`, `bull`, `summarize`, `judge`. Each is an async function that
  receives the state and returns a *partial* state update.
- **Edges:** `START → bear`, `bear → bull` are static. The `bull` and `summarize`
  nodes are followed by **conditional edges** (routing functions).
- **Checkpointer:** `InMemorySaver` (dev) or `AsyncPostgresSaver` (prod) serializes
  every state snapshot, enabling `thread_id` isolation, resume, and HITL.

---

## 3. LangGraph state: `DebateState` and reducers

State is a `TypedDict`. The key LangGraph concept here is the **reducer**: a function
that defines *how* a node's update merges into the existing state. Without a reducer,
a list field returned by a node *overwrites* the prior value; with a reducer, it
*appends* (or merges, or deletes).

```python
# src/bear_bull_debate/state.py
import operator
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class DebateState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]   # append + dedupe + RemoveMessage
    round: int                                              # scalar: last-write-wins
    company: str
    summary: str
    tool_outputs: Annotated[list[str], operator.add]        # append (list concat)
    max_rounds: int
```

| Field | Reducer | Why |
|-------|---------|-----|
| `messages` | `add_messages` | Appends new messages, dedupes by `id`, and *processes `RemoveMessage`* to physically delete old messages. This is the LangGraph-blessed message-list reducer. |
| `tool_outputs` | `operator.add` | Appends (concatenates) so each node's tool logs accumulate instead of overwriting. |
| `round`, `company`, `summary`, `max_rounds` | (default) | Scalars; last-write-wins is correct. |

`max_rounds` lives in state (seeded from the request) so that **routing depends only on
state** — a per-request override works without a global.

---

## 4. LangChain building blocks

The model/message/tool layer uses LangChain primitives directly:

| LangChain primitive | Where used | Role |
|---------------------|-----------|------|
| `BaseMessage` subclasses (`SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`) | `nodes.py`, `api.py` | The message protocol passed to/from the model. |
| `RemoveMessage` | `nodes.py` (summarize) | A marker message the `add_messages` reducer interprets as "delete this id". |
| `ToolException` | `tools.py`, `nodes.py` | Standard exception for tool failures → graceful degradation. |
| `@tool` | `tools.py` | Declares the mock tools with a JSON schema the model can call. |
| `ChatOpenAI` | `llm.py` | The chat model; supports `base_url`/`api_key` for OpenAI-compatible providers. |
| `bind_tools(...)` | `graph.py` | Attaches tool schemas to the Bear/Bull models so they emit `tool_calls`. |
| `FakeMessagesListChatModel` | tests | Deterministic model for TDD without network/API keys. |
| `ConsoleCallbackHandler` | `llm.py` (VERBOSE) | Verbose token/tool logging. |

### 4.1 Tools (`tools.py`)

Three mock tools (`get_stock_price`, `get_financials`, `get_news_sentiment`) are declared
with the `@tool` decorator, which introspects the function signature and docstring into a
JSON schema. They raise `ToolException` on invalid input, which the researcher node
catches and converts into a recoverable `ToolMessage`.

### 4.2 Model factory (`llm.py`)

`make_llm(model)` returns a `ChatOpenAI` configured for the given model. It explicitly
reads `OPENAI_BASE_URL` and `OPENAI_API_KEY` so the same code path works against
OpenAI-compatible providers (DeepSeek, Qwen/DashScope) and inside Colab (where the key
is injected programmatically rather than via a `.env` file).

---

## 5. Nodes (the agent logic)

### 5.1 Bear/Bull researcher — the tool-calling agent loop

Each researcher is the *same* node factory parameterized by `role`:

```python
def make_researcher_node(role, llm, tools, settings) -> Callable:
    ...
```

The loop implements the canonical **agentic tool-calling pattern** manually (instead of
LangGraph's `ToolNode`) to retain fine-grained control over error handling:

```text
build context (system prompt + summary seed + last N history messages)
  └─▶ call LLM (with retry)
        ├─ response has tool_calls?  ── yes ──▶ execute each tool, append ToolMessage, loop
        └─ no tool_calls ── break (the AIMessage is the researcher's argument)
```

Key engineering details:

- **Retry** (`ainvoke_with_retry`) wraps *only* the LLM call with `tenacity`
  (3 attempts, exponential backoff) on transient errors (`APIConnectionError`,
  `APITimeoutError`, `RateLimitError`). A whole-node retry would recompute state; retrying
  only the network hop avoids that.
- **Graceful tool failure:** `ToolException` → a `ToolMessage` telling the model to fix
  its parameters; unknown tool (`KeyError`) and any other error (`Exception`, noqa) also
  degrade to a `ToolMessage` rather than crashing the graph.
- **`ToolMessage` contract:** every `ToolMessage` carries both `tool_call_id` and `name`
  (omitting `name` is a common OpenAI-API 400 failure).
- **Hard cap:** the loop runs at most `max_tool_rounds` iterations. If the model keeps
  requesting tools, the node forces one final synthesis call so the turn ends with an
  *argument*, not a dangling tool result.
- **Round accounting:** only the `bull` node increments `round`, so `round` == completed
  debate cycles (bear→bull).

### 5.2 Summarize — context compaction with `RemoveMessage`

Long debates would blow the context window. The summarize node compresses the oldest
messages into a running `summary` and *physically frees* them:

```python
old = messages[:-settings.history_window]           # everything except the last N
summary = await llm(...)                             # condense `old` into text
return {"summary": summary, "messages": [RemoveMessage(id=m.id) for m in old if m.id]}
```

Returning `RemoveMessage(id=...)` is the correct LangGraph idiom — the `add_messages`
reducer sees these markers and deletes the matching messages from state (releasing
memory), which a naive `state["messages"] = state["messages"][-N:]` would not do safely
inside a reducer.

### 5.3 Judge — bias-resistant final report

The Judge receives `summary` + recent messages + company. Its system prompt forces a
point-by-point comparison (Bear's case → Bull's case → head-to-head → verdict) *before*
a conclusion, countering recency and length bias. It emits a structured Markdown report
(`## Verdict`, `## Bear's Case`, `## Bull's Case`, `## Head-to-Head`,
`## Recommendation`, `## Confidence`).

---

## 6. Routing (conditional edges)

Routing is abstracted into pure factory functions that close over `Settings`:

```python
def make_route_after_bull(settings):
    def route(state):
        if len(state["messages"]) > settings.message_threshold:
            return "summarize"          # priority 1: context getting long
        if state["round"] >= state["max_rounds"]:
            return "judge"              # priority 2: debate done
        return "bear"                   # priority 3: continue
    return route
```

`route_after_summarize` then decides `judge` (if done) or `bear` (continue). Keeping the
router pure (depends only on state) makes every branch unit-testable without a model.

---

## 7. Persistence & checkpointing

```python
# checkpointer.py
async def make_checkpointer(uri):
    if not uri:
        return InMemorySaver()
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    saver = AsyncPostgresSaver.from_conn_string(uri)
    await saver.setup()
    return saver
```

- **`thread_id` isolation:** `config = {"configurable": {"thread_id": ...}}` scopes every
  checkpoint to a conversation.
- **Resume / HITL:** `builder.compile(checkpointer=..., interrupt_before=["judge"])`
  pauses before the judge; a second `ainvoke(None, config)` resumes from the checkpoint.
  This is the foundation for human-in-the-loop (a human could edit messages / inject a
  `HumanMessage` before resuming).

---

## 8. API layer (`api.py`)

FastAPI + Pydantic v2:

- `DebateRequest` validates `company` against `^[a-zA-Z0-9 .\-]+$` (literal space, not
  `\s`, so newlines cannot smuggle prompt-injection payloads) and `max_rounds ∈ [1,5]`.
- `create_app` builds the graph in a **lifespan** (not at import), opens/closes the
  checkpointer, and exposes `POST /api/v1/debate` + `GET /healthz`.
- `run_debate` invokes the graph and returns `{thread_id, final_report, tool_logs}`.

A separate `runner.py` provides `run_debate()` / `run_debate_async()` for non-HTTP use
(scripts, REPL, Colab).

---

## 9. Observability

- Each node is decorated with `@traceable(name=...)` via a guarded `trace()` wrapper that
  no-ops when `langsmith` is absent. Set `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY`
  for per-node token/latency/tool traces.
- `VERBOSE=1` adds `ConsoleCallbackHandler` for local token/tool logging.

---

## 10. Data flow (end-to-end)

```
POST /api/v1/debate {company, max_rounds}
  → DebateRequest validated
  → initial_state: messages=[HumanMessage(seed)], round=0, max_rounds
  → graph.ainvoke(state, {thread_id})
       bear  : (tool loop) → AIMessage (bear argument)
       bull  : (tool loop) → AIMessage (bull argument), round++
       router: threshold? → summarize → (router) → bear | judge
               else round>=max_rounds? → judge
       judge : summary + recent messages → structured report
  → result.messages[-1] is the judge's report
  → {"thread_id", "final_report", "tool_logs"}
```

---

## 11. Error handling & safety

| Concern | Mechanism |
|---------|-----------|
| Transient LLM errors | `tenacity` retry (3×, exponential backoff) on `openai` transient exceptions |
| Tool failure | `ToolException`/`KeyError`/generic caught → recoverable `ToolMessage` |
| Pathological tool-call loop | `max_tool_rounds` cap + forced final synthesis |
| Context overflow | `summarize` node + `RemoveMessage` (physical memory release) |
| Prompt injection | strict `company` regex (no newlines); company treated as data in user message, not trusted system content |
| State overwrite bug | `operator.add` / `add_messages` reducers |
| Missing Postgres extra | lazy import → clear `RuntimeError` with install hint |

---

## 12. Testing strategy

- **Unit** — reducers, tools, retry, each node (with `FakeMessagesListChatModel`), router.
- **Graph integration** — full runs (2 rounds → judge; threshold → summarize; resume
  from `interrupt_before`).
- **API** — Pydantic validation, endpoint smoke via `TestClient` + `StubGraph`.
- **Runner** — sync/async entry points via a monkeypatched graph.

Run: `uv run pytest -q` (44 tests at the time of writing).

---

## 13. Configuration

| Env var | Default | Purpose |
|---------|---------|---------|
| `BEAR_MODEL` / `BULL_MODEL` | `gpt-4o-mini` | Researcher models |
| `JUDGE_MODEL` | `gpt-4o` | Judge model (strong reasoning) |
| `SUMMARY_MODEL` | `gpt-4o-mini` | Summarizer model |
| `MAX_ROUNDS` | `2` | Default rounds when request omits it |
| `HISTORY_WINDOW` | `4` | Recent messages passed to the LLM |
| `MESSAGE_THRESHOLD` | `12` | Trigger summarize |
| `MAX_TOOL_ROUNDS` | `5` | Tool-call loop cap |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | — | Credentials / compatible endpoint |
| `DATABASE_URL` | (empty) | Postgres checkpointer (empty → in-memory) |

---

## 14. Future work (V2+)

- **Real tools:** swap mock tools for `yfinance` / `Tavily` / `Alpha Vantage`; adopt
  LangGraph's `ToolNode` for parallel tool calls and automatic retry.
- **Human-in-the-loop:** insert a dynamic `interrupt()` before the Judge and let a human
  edit messages before resuming.
- **Reflection:** add a `Critic` node that challenges the Judge's report, forcing a
  second pass.
- **Multimodal:** return charts as `ImageMessage` content for GPT-4o joint analysis.
- **Multi-agent topology:** insert a `Macro_Analyst` between Bear and Bull.
