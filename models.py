from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class WorkerAnswer(BaseModel):
    question_id: int
    question: str
    answer: str
    has_error: bool = Field(exclude=True)
    error_type: Optional[Literal["wrong_value", "wrong_inference", "omission"]] = Field(
        default=None, exclude=True
    )
    correct_answer: str = Field(exclude=True)
    relevant_field: str = Field(exclude=True)


class OversightObservation(BaseModel):
    source_json: dict[str, Any]
    questions: list[str]
    worker_answers: list[str]
    step_number: int
    flags_used: int
    flags_remaining: int
    episode_id: str
    done: bool
    message: str


class OversightAction(BaseModel):
    action_type: Literal["approve", "flag"]
    question_id: int
    error_type: Optional[Literal["wrong_value", "wrong_inference", "omission"]] = None
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


# EpisodeState is a plain dict — mutated freely by the environment each step.
EpisodeState = dict[str, Any]


def make_episode_state(
    episode_id: str,
    task_id: str,
    source_json: dict[str, Any],
    worker_answers: list[WorkerAnswer],
) -> EpisodeState:
    return {
        "episode_id": episode_id,
        "task_id": task_id,
        "source_json": source_json,
        "worker_answers": worker_answers,
        "agent_decisions": [],
        "flags_used": 0,
        "step_number": 0,
        "total_reward": 0.0,
    }
