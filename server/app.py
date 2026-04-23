"""FastAPI server exposing OpenEnv-compliant endpoints for OversightArena."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from environment import OversightEnvironment
from models import OversightAction, OversightObservation

app = FastAPI(title="OversightArena", version="1.0.0")
env = OversightEnvironment()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy"


class ResetResponse(BaseModel):
    observation: OversightObservation


class StepRequest(BaseModel):
    action: OversightAction


class StepResponse(BaseModel):
    observation: OversightObservation
    reward: float
    done: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": "OversightArena"}


@app.get("/observation_space")
def observation_space() -> dict[str, Any]:
    return env.get_observation_space()


@app.get("/action_space")
def action_space() -> dict[str, Any]:
    return env.get_action_space()


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest) -> ResetResponse:
    try:
        obs = env.reset(task_id=request.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return ResetResponse(observation=obs)


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    try:
        obs, reward = env.step(request.action)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return StepResponse(observation=obs, reward=reward, done=obs.done)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
