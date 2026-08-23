# Bull-Bear Debate Stock Analysis System

A LangGraph-based multi-round Bull vs Bear debate with a Judge that emits a structured,
bias-resistant investment report. Tools are mocked in V1; the debate mechanism and agent
coordination are the focus.

See [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md) for the full design (LangChain +
LangGraph usage).

## Quickstart (API server)

```bash
uv sync --extra dev
cp .env.example .env   # set OPENAI_API_KEY
uv run uvicorn bear_bull_debate.api:app --env-file .env --reload
curl -X POST http://127.0.0.1:8000/api/v1/debate \
  -H 'Content-Type: application/json' \
  -d '{"company": "AAPL", "max_rounds": 2}'
```

## Quickstart (script / notebook, no server)

```python
import nest_asyncio; nest_asyncio.apply()  # only needed inside Jupyter/Colab
from bear_bull_debate.runner import run_debate

result = run_debate("AAPL", max_rounds=2)
print(result["final_report"])
print(result["tool_logs"])
```

## Google Colab

Open [`notebooks/bear_bull_debate_colab.ipynb`](notebooks/bear_bull_debate_colab.ipynb)
in Colab. It installs dependencies, uploads the source as a ZIP, loads your
`OPENAI_API_KEY` from Colab Secrets (or prompts for it), and runs a debate.

To prepare the ZIP locally:

```bash
python tools/build_bear_bull_debate_zip.py
```

(or equivalently `zip -r bear_bull_debate_src.zip src/bear_bull_debate`). Then upload
`bear_bull_debate_src.zip` when the notebook prompts (or zip the whole project — the
notebook auto-locates the package).

OpenAI-compatible providers (DeepSeek / Qwen) are supported via `OPENAI_BASE_URL`.
**Important:** your API key must match the provider. A DeepSeek key sent to OpenAI's
default endpoint is rejected with a 401 (`Incorrect API key provided`). Set all of:

```bash
export OPENAI_API_KEY=sk-your-deepseek-key
export OPENAI_BASE_URL=https://api.deepseek.com
export BEAR_MODEL=deepseek-chat
export BULL_MODEL=deepseek-chat
export JUDGE_MODEL=deepseek-chat
export SUMMARY_MODEL=deepseek-chat
```

In the Colab notebook, just pick the provider in cell 3 (OpenAI / DeepSeek /
Qwen DashScope) — it sets `OPENAI_BASE_URL` and the model names automatically.

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
