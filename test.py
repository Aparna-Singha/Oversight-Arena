"""
Integration tests for OversightArena.
Requires the server to be running at http://localhost:8000.

Usage:
    python test.py
"""
from __future__ import annotations

import sys

import requests

BASE = "http://localhost:8000"


def test_health() -> None:
    r = requests.get(f"{BASE}/health", timeout=5)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    assert r.json()["status"] == "ok"
    print("PASS: /health")


def test_task(task_id: str) -> None:
    print(f"\n=== Testing {task_id} ===")

    # Reset
    r = requests.post(f"{BASE}/reset", json={"task_id": task_id}, timeout=5)
    assert r.status_code == 200, f"/reset returned {r.status_code}: {r.text}"
    obs = r.json()["observation"]

    print(f"Questions      : {len(obs['questions'])}")
    print(f"Step           : {obs['step_number']}")
    print(f"Flags remaining: {obs['flags_remaining']}")

    assert len(obs["questions"]) == 5,        "Expected 5 questions"
    assert obs["step_number"] == 0,            "step_number should start at 0"
    assert obs["flags_remaining"] == 5,        "flags_remaining should start at 5"
    assert obs["done"] is False,               "done should be False at start"
    assert "has_error" not in str(obs["worker_answers"]), \
        "Hidden fields must not be visible in worker_answers"

    # Take 5 actions â€” flag q0, approve the rest
    result: dict = {}
    for i in range(5):
        action = {
            "action": {
                "action_type": "flag" if i == 0 else "approve",
                "question_id": i,
                "error_type": "wrong_value" if i == 0 else None,
                "reasoning": f"Checking field {i}",
                "confidence": 0.8,
            }
        }
        r = requests.post(f"{BASE}/step", json=action, timeout=5)
        assert r.status_code == 200, f"/step {i} returned {r.status_code}: {r.text}"
        result = r.json()
        print(f"  Step {i}: reward={result['reward']:.2f}  done={result['done']}")

        assert "reward" in result, "Step response missing 'reward'"
        assert "done"   in result, "Step response missing 'done'"
        assert 0.0 <= result["reward"] <= 1.0, \
            f"Reward out of range: {result['reward']}"

    assert result["done"] is True, "Episode should be done after 5 steps"
    print(f"PASS: {task_id}")


def test_invalid_task_id() -> None:
    r = requests.post(f"{BASE}/reset", json={"task_id": "impossible"}, timeout=5)
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print("PASS: invalid task_id returns 422")


def test_step_without_reset() -> None:
    # Fresh server state is shared â€” only safe to run this in isolation.
    # Skip if we can't guarantee a clean state.
    print("SKIP: test_step_without_reset (requires isolated server state)")


def main() -> None:
    try:
        requests.get(f"{BASE}/health", timeout=3)
    except Exception:
        print(f"ERROR: server not reachable at {BASE} â€” start it first.")
        sys.exit(1)

    test_health()
    for task in ["easy", "medium", "hard", "expert"]:
        test_task(task)
    test_invalid_task_id()

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
