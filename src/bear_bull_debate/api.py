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
