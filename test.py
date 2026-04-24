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
    reset_body = r.json()
    obs = reset_body["observation"]
    episode_id = reset_body["episode_id"]

    print(f"Questions      : {len(obs['questions'])}")
    print(f"Step           : {obs['step_number']}")
    print(f"Flags remaining: {obs['flags_remaining']}")
    print(f"Episode ID     : {episode_id}")

    assert len(obs["questions"]) == 5,        "Expected 5 questions"
    assert obs["step_number"] == 0,            "step_number should start at 0"
    assert obs["flags_remaining"] == 5,        "flags_remaining should start at 5"
    assert obs["done"] is False,               "done should be False at start"
    assert episode_id == obs["episode_id"],    "episode_id must match in body and observation"
    assert "has_error" not in str(obs["worker_answers"]), \
        "Hidden fields must not be visible in worker_answers"

    # Take 5 actions -- flag q0, approve the rest
    result: dict = {}
    for i in range(5):
        action = {
            "episode_id": episode_id,
            "action": {
                "action_type": "flag" if i == 0 else "approve",
                "question_id": i,
                "error_type": "wrong_value" if i == 0 else None,
                "reasoning": f"Checking field {i}",
                "confidence": 0.8,
            },
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


def test_invalid_episode_id() -> None:
    action = {
        "episode_id": "00000000-0000-0000-0000-000000000000",
        "action": {
            "action_type": "approve",
            "question_id": 0,
            "error_type": None,
            "reasoning": "looks fine",
            "confidence": 0.5,
        },
    }
    r = requests.post(f"{BASE}/step", json=action, timeout=5)
    assert r.status_code == 400, f"Expected 400, got {r.status_code}: {r.text}"
    print("PASS: invalid episode_id returns 400")


def test_parallel_episodes() -> None:
    # Start two episodes simultaneously and verify their state is isolated.
    r1 = requests.post(f"{BASE}/reset", json={"task_id": "easy"}, timeout=5)
    r2 = requests.post(f"{BASE}/reset", json={"task_id": "easy"}, timeout=5)
    assert r1.status_code == 200 and r2.status_code == 200

    eid1 = r1.json()["episode_id"]
    eid2 = r2.json()["episode_id"]
    assert eid1 != eid2, "Two resets must produce distinct episode IDs"

    def make_step(episode_id: str, q: int) -> dict:
        return {
            "episode_id": episode_id,
            "action": {
                "action_type": "flag",
                "question_id": q,
                "error_type": "wrong_value",
                "reasoning": f"episode {episode_id[:8]} step {q}",
                "confidence": 0.9,
            },
        }

    # Advance episode 1 once; episode 2 should still be at step 0
    s1 = requests.post(f"{BASE}/step", json=make_step(eid1, 0), timeout=5)
    assert s1.status_code == 200, f"Step on ep1 failed: {s1.text}"

    # Step episode 2 independently
    s2 = requests.post(f"{BASE}/step", json=make_step(eid2, 0), timeout=5)
    assert s2.status_code == 200, f"Step on ep2 failed: {s2.text}"

    obs1 = s1.json()["observation"]
    obs2 = s2.json()["observation"]
    assert obs1["episode_id"] == eid1, "ep1 obs has wrong episode_id"
    assert obs2["episode_id"] == eid2, "ep2 obs has wrong episode_id"
    assert obs1["step_number"] == obs2["step_number"] == 1, \
        "Both episodes should be at step 1 after one step each"

    # Exhaust both episodes cleanly (4 more steps each)
    for eid in [eid1, eid2]:
        for q in range(1, 5):
            requests.post(f"{BASE}/step", json=make_step(eid, q), timeout=5)

    print("PASS: parallel episodes maintain isolated state")


def test_step_without_reset() -> None:
    print("SKIP: test_step_without_reset (requires isolated server state)")


def main() -> None:
    try:
        requests.get(f"{BASE}/health", timeout=3)
    except Exception:
        print(f"ERROR: server not reachable at {BASE} -- start it first.")
        sys.exit(1)

    test_health()
    for task in ["easy", "medium", "hard", "expert"]:
        test_task(task)
    test_invalid_task_id()
    test_invalid_episode_id()
    test_parallel_episodes()

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
