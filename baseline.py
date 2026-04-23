"""
Baseline agent for OversightArena.

Implements three strategies:
  - random:    flag/approve with 50/50 probability
  - always_flag: flag everything (recall-maximising, precision-destroying)
  - heuristic: simple keyword/numeric heuristics on the answer text

Usage:
    python baseline.py --strategy heuristic --episodes 10
"""
from __future__ import annotations

import argparse
import random
import re
import time

import requests

SERVER = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def random_strategy(obs: dict, rng: random.Random) -> dict:
    idx = obs["step_number"] % len(obs["questions"])
    action_type = rng.choice(["approve", "flag"])
    return {
        "action_type": action_type,
        "question_id": idx,
        "error_type": rng.choice(["wrong_value", "wrong_inference", "omission"])
        if action_type == "flag"
        else None,
        "reasoning": "Random baseline decision.",
        "confidence": round(rng.uniform(0.4, 0.9), 2),
    }


def always_flag_strategy(obs: dict, _rng: random.Random) -> dict:
    idx = obs["step_number"] % len(obs["questions"])
    return {
        "action_type": "flag",
        "question_id": idx,
        "error_type": "wrong_value",
        "reasoning": "Always-flag baseline.",
        "confidence": 0.5,
    }


def heuristic_strategy(obs: dict, _rng: random.Random) -> dict:
    idx = obs["step_number"] % len(obs["questions"])
    answer = obs["worker_answers"][idx]

    suspicious = False
    error_type = "wrong_value"

    if re.search(r"no information|not available|unknown", answer, re.I):
        suspicious = True
        error_type = "omission"
    elif re.search(r"\bestimated\b|\bapprox\b", answer, re.I):
        suspicious = True
        error_type = "wrong_inference"
    elif re.search(r"\$[\d,]+\.\d{2}", answer):
        nums = re.findall(r"[\d,]+\.?\d*", answer.replace(",", ""))
        if any(float(n) <= 0 for n in nums if n):
            suspicious = True
            error_type = "wrong_value"

    return {
        "action_type": "flag" if suspicious else "approve",
        "question_id": idx,
        "error_type": error_type if suspicious else None,
        "reasoning": "Heuristic: suspicious pattern detected." if suspicious else "Heuristic: answer looks clean.",
        "confidence": 0.7 if suspicious else 0.8,
    }


STRATEGIES = {
    "random": random_strategy,
    "always_flag": always_flag_strategy,
    "heuristic": heuristic_strategy,
}


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------

def run_episode(strategy_fn, rng: random.Random, verbose: bool = True) -> dict:
    obs = requests.post(f"{SERVER}/reset", json={}).json()
    episode_id = obs["episode_id"]
    total_reward = 0.0

    while not obs["done"]:
        action = strategy_fn(obs, rng)
        resp = requests.post(
            f"{SERVER}/step",
            json={"episode_id": episode_id, "action": action},
        ).json()
        obs = resp["observation"]
        total_reward += resp["reward"]
        if verbose:
            print(
                f"  step={obs['step_number']}  q={action['question_id']}"
                f"  action={action['action_type']:<7}  reward={resp['reward']:+.1f}"
                f"  {resp['info'].get('outcome', '')}"
            )

    summary = resp["info"].get("summary", {})
    return {"total_reward": total_reward, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="OversightArena baseline agent")
    parser.add_argument("--strategy", choices=list(STRATEGIES), default="heuristic")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    strategy_fn = STRATEGIES[args.strategy]

    rewards = []
    for ep in range(1, args.episodes + 1):
        print(f"\n=== Episode {ep}/{args.episodes} â€” strategy={args.strategy} ===")
        result = run_episode(strategy_fn, rng, verbose=not args.quiet)
        rewards.append(result["total_reward"])
        s = result["summary"]
        print(
            f"  total_reward={result['total_reward']:+.1f}  "
            f"f1={s.get('f1', 0):.3f}  "
            f"precision={s.get('precision', 0):.3f}  "
            f"recall={s.get('recall', 0):.3f}"
        )

    print(f"\n--- {args.episodes}-episode summary ---")
    print(f"mean_reward = {sum(rewards)/len(rewards):.2f}")
    print(f"min_reward  = {min(rewards):.2f}")
    print(f"max_reward  = {max(rewards):.2f}")


if __name__ == "__main__":
    main()
