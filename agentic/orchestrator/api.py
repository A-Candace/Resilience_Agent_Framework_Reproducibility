from uuid import uuid4
from fastapi import FastAPI
from pydantic import BaseModel, Field
from agentic.common.observability import configure_observability
from agentic.orchestrator.service import run_agent

app = FastAPI(title="NYC Resilience Agent Orchestrator", version="1.0.0")


class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    session_id: str | None = None


@app.on_event("startup")
def startup() -> None:
    configure_observability()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "orchestrator"}


@app.post("/v1/agent")
async def agent(request: AgentRequest) -> dict:
    return await run_agent(request.message, request.session_id or str(uuid4()))
