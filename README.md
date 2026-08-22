# Bull-Bear Debate Stock Analysis System

A LangGraph-based multi-round Bull vs Bear debate with a Judge that emits a structured,
bias-resistant investment report. Tools are mocked in V1; the debate mechanism and agent
coordination are the focus.

## Quickstart

```bash
uv sync --extra dev
cp .env.example .env   # set OPENAI_API_KEY
uv run uvicorn bear_bull_debate.api:app --env-file .env --reload
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
uv run uvicorn bear_bull_debate.api:app --env-file .env --reload
```

The app lazily imports `AsyncPostgresSaver` only when `DATABASE_URL` is set and runs
`checkpointer.setup()` on startup to create tables.

## Observability

Set `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` for LangSmith traces (each node is
decorated with `@traceable`). Set `VERBOSE=1` for console callback output.
