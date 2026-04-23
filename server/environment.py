"""Core RL environment logic for OversightArena."""
from __future__ import annotations

import json
import os
import random
import sys
import uuid
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    EpisodeState,
    OversightAction,
    OversightObservation,
    WorkerAnswer,
    make_episode_state,
)

try:
    from grader import grade_step
except ImportError:
    def grade_step(action: OversightAction, answer: WorkerAnswer) -> float:  # type: ignore[misc]
        return 0.0

FLAG_BUDGET = 5

_TASKS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "oversight_tasks.json",
)


def _load_tasks(path: str) -> dict[str, list[dict[str, Any]]]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _reconstruct_worker_answers(raw: list[dict[str, Any]]) -> list[WorkerAnswer]:
    return [WorkerAnswer(**wa) for wa in raw]


class OversightEnvironment:
    def __init__(self) -> None:
        self._tasks: dict[str, list[dict[str, Any]]] = _load_tasks(_TASKS_PATH)
        self._state: EpisodeState | None = None
        self._rng = random.Random()

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "easy") -> OversightObservation:
        if task_id not in self._tasks:
            raise ValueError(
                f"Unknown difficulty: {task_id!r}. "
                f"Choose from {list(self._tasks)}"
            )

        task = self._rng.choice(self._tasks[task_id])
        worker_answers = _reconstruct_worker_answers(task["worker_answers"])
        episode_id = str(uuid.uuid4())

        self._state = make_episode_state(
            episode_id=episode_id,
            task_id=task_id,
            source_json=task["source_json"],
            worker_answers=worker_answers,
        )

        return self._build_observation(done=False)

    def step(self, action: OversightAction) -> tuple[OversightObservation, float]:
        if self._state is None:
            raise ValueError("No active episode â€” call reset() first.")

        if action.question_id < 0 or action.question_id > 4:
            raise ValueError(
                f"question_id must be 0â€“4, got {action.question_id}"
            )

        state = self._state

        if action.action_type == "flag":
            state["flags_used"] += 1

        answer = state["worker_answers"][action.question_id]
        reward = grade_step(action, answer)
        state["total_reward"] += reward

        state["agent_decisions"].append(
            {
                "step": state["step_number"],
                "question_id": action.question_id,
                "action_type": action.action_type,
                "error_type_claimed": action.error_type,
                "reward": reward,
                "confidence": action.confidence,
                "reasoning": action.reasoning,
            }
        )

        state["step_number"] += 1

        flags_remaining = FLAG_BUDGET - state["flags_used"]
        done = (state["step_number"] >= 5) or (flags_remaining <= 0)

        return self._build_observation(done=done), reward

    def get_observation_space(self) -> dict[str, Any]:
        return {
            "type": "object",
            "fields": {
                "source_json": "dict",
                "questions": "List[str]",
                "worker_answers": "List[str]",
                "step_number": "int",
                "flags_used": "int",
                "flags_remaining": "int",
                "episode_id": "str",
                "done": "bool",
                "message": "str",
            },
        }

    def get_action_space(self) -> dict[str, Any]:
        return {
            "type": "object",
            "fields": {
                "action_type": "Literal['approve', 'flag']",
                "question_id": "int (0â€“4)",
                "error_type": "Optional[Literal['wrong_value','wrong_inference','omission']]",
                "reasoning": "str",
                "confidence": "float [0.0, 1.0]",
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self, done: bool) -> OversightObservation:
        state = self._state
        answers: list[WorkerAnswer] = state["worker_answers"]
        flags_used = state["flags_used"]
        step = state["step_number"]
        task_id = state["task_id"]

        return OversightObservation(
            source_json=state["source_json"],
            questions=[wa.question for wa in answers],
            worker_answers=[wa.answer for wa in answers],
            step_number=step,
            flags_used=flags_used,
            flags_remaining=FLAG_BUDGET - flags_used,
            episode_id=state["episode_id"],
            done=done,
            message=f"Step {step} | Flags used: {flags_used}/5 | Task: {task_id}",
        )
