"""
Data integrity checks for data/oversight_tasks.json.

Usage:
    python verify_data.py
    python verify_data.py --path path/to/oversight_tasks.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def check_error_indices_match_has_error(dataset: dict) -> list[str]:
    """CHECK 1: error_indices must exactly match question_ids where has_error=True."""
    failures = []
    for diff, tasks in dataset.items():
        for task in tasks:
            tid = task["task_id"][:8]
            flagged_ids = {wa["question_id"] for wa in task["worker_answers"] if wa["has_error"]}
            declared = set(task["error_indices"])
            if flagged_ids != declared:
                failures.append(
                    f"  FAIL [{diff}] task={tid}: error_indices={sorted(declared)}"
                    f" but has_error ids={sorted(flagged_ids)}"
                )
    return failures


def check_num_errors_matches_len(dataset: dict) -> list[str]:
    """CHECK 2: num_errors must equal len(error_indices)."""
    failures = []
    for diff, tasks in dataset.items():
        for task in tasks:
            tid = task["task_id"][:8]
            declared = task["num_errors"]
            actual = len(task["error_indices"])
            if declared != actual:
                failures.append(
                    f"  FAIL [{diff}] task={tid}: num_errors={declared}"
                    f" but len(error_indices)={actual}"
                )
    return failures


def check_no_error_with_same_answer(dataset: dict) -> list[str]:
    """CHECK 3: No entry where has_error=True AND answer==correct_answer."""
    failures = []
    for diff, tasks in dataset.items():
        for task in tasks:
            tid = task["task_id"][:8]
            for wa in task["worker_answers"]:
                if wa["has_error"] and wa["answer"] == wa["correct_answer"]:
                    failures.append(
                        f"  FAIL [{diff}] task={tid} q={wa['question_id']}:"
                        f" has_error=True but answer==correct_answer={wa['answer']!r}"
                    )
    return failures


def check_all_difficulties_present(dataset: dict) -> list[str]:
    """CHECK 4: All four difficulty levels must have at least one entry."""
    required = {"easy", "medium", "hard", "expert"}
    failures = []
    for level in required:
        if level not in dataset or len(dataset[level]) == 0:
            failures.append(f"  FAIL: difficulty level {level!r} missing or empty")
    return failures


def check_expert_distractors(dataset: dict) -> list[str]:
    """CHECK 5: Each expert entry must have >=2 has_error=False answers where answer!=correct_answer."""
    warnings = []
    for task in dataset.get("expert", []):
        tid = task["task_id"][:8]
        distractors = [
            wa for wa in task["worker_answers"]
            if not wa["has_error"] and wa["answer"] != wa["correct_answer"]
        ]
        if len(distractors) < 2:
            warnings.append(
                f"  WARNING [expert] task={tid}: only {len(distractors)} distractor(s) found"
                f" (expected >=2)"
            )
    return warnings


def main(path: str) -> None:
    if not os.path.exists(path):
        print(f"ERROR: file not found: {path}")
        sys.exit(1)

    dataset = load(path)
    total = sum(len(v) for v in dataset.values())
    print(f"Loaded {total} entries from {path}\n")

    checks = [
        ("CHECK 1 â€” error_indices match has_error fields", check_error_indices_match_has_error),
        ("CHECK 2 â€” num_errors matches len(error_indices)",  check_num_errors_matches_len),
        ("CHECK 3 â€” no has_error=True with answer==correct_answer", check_no_error_with_same_answer),
        ("CHECK 4 â€” all difficulty levels present",           check_all_difficulties_present),
        ("CHECK 5 â€” expert entries have >=2 distractors",    check_expert_distractors),
    ]

    passed = 0
    failed = 0

    for label, fn in checks:
        issues = fn(dataset)
        is_warning_check = "distractors" in label
        if not issues:
            print(f"PASS  {label}")
            passed += 1
        else:
            if is_warning_check:
                print(f"WARN  {label}")
                for msg in issues:
                    print(msg)
                passed += 1
            else:
                print(f"FAIL  {label}")
                for msg in issues:
                    print(msg)
                failed += 1

    print(f"\nTotal entries : {total}")
    print(f"Checks passed : {passed}")
    print(f"Checks failed : {failed}")

    if failed == 0:
        print("\nDATA VERIFIED â€” ALL CHECKS PASSED")
    else:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify OversightArena dataset integrity")
    parser.add_argument(
        "--path",
        default=os.path.join(os.path.dirname(__file__), "data", "oversight_tasks.json"),
        help="Path to oversight_tasks.json",
    )
    args = parser.parse_args()
    main(args.path)
