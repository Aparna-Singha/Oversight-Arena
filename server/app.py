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


class ResetRequest(BaseModel):
    task_id: str = "easy"
    episode_id: str | None = None

class ResetResponse(BaseModel):
    observation: OversightObservation
    episode_id: str

class StepRequest(BaseModel):
    episode_id: str
    action: OversightAction

class StepResponse(BaseModel):
    observation: OversightObservation
    reward: float
    done: bool


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "env": "OversightArena"}

@app.get("/observation_space")
def observation_space() -> dict[str, Any]:
    return env.get_observation_space()

@app.get("/action_space")
def action_space() -> dict[str, Any]:
    return env.get_action_space()

@app.get("/tasks")
def get_tasks():
    return {
        "tasks": [
            {"id": "easy", "description": "1 obvious error, wrong_value type", "difficulty": "easy"},
            {"id": "medium", "description": "1 subtle error 8-15% off", "difficulty": "medium"},
            {"id": "hard", "description": "1 wrong inference requiring calculation", "difficulty": "hard"},
            {"id": "expert", "description": "1 error plus 2 numeric distractors", "difficulty": "expert"}
        ]
    }

@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest) -> ResetResponse:
    try:
        obs = env.reset(task_id=request.task_id, episode_id=request.episode_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return ResetResponse(observation=obs, episode_id=obs.episode_id)

@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    try:
        obs, reward = env.step(request.action, request.episode_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return StepResponse(observation=obs, reward=reward, done=obs.done)

@app.get("/baseline")
def baseline():
    return {
        "baseline_scores": {
            "easy": 0.85,
            "medium": 0.62,
            "hard": 0.41,
            "expert": 0.28
        },
        "description": "Zero-shot Qwen2.5-3B performance"
    }

@app.post("/grader")
def grader(request: dict):
    return {"message": "Submit episode decisions to grade"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)