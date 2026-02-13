#!/usr/bin/env python3
"""
Merge partial task_comp_l1 JSON summaries (split across nodes) into a single
per-seed summary file.

Usage:
    python merge_task_comp_results.py \
        --summary_dir /mnt/beegfs/a.cardamone7/outputs/logs/summary/task_comp_l1 \
        --seeds 0 1 2 \
        --num_parts 2

Expects files named:  tinyvla_seed{SEED}_part{PART}.json
Produces:             tinyvla_seed{SEED}_merged.json
"""

import argparse
import json
import os
import sys


def merge_seed(summary_dir: str, seed: int, num_parts: int) -> dict:
    merged_task_results = {}
    total_episodes = 0
    total_successes = 0

    for part in range(num_parts):
        filename = f"tinyvla_seed{seed}_part{part}.json"
        filepath = os.path.join(summary_dir, filename)

        if not os.path.exists(filepath):
            print(f"  [WARNING] Missing: {filepath}")
            continue

        with open(filepath, "r") as f:
            data = json.load(f)

        # Merge task-level results
        for task_name, result in data.get("task_results", {}).items():
            if task_name in merged_task_results:
                print(f"  [WARNING] Duplicate task '{task_name}' in part {part}, overwriting")
            merged_task_results[task_name] = result
            total_episodes += result["episodes"]
            total_successes += int(result["success_rate"] * result["episodes"])

    overall_sr = total_successes / total_episodes if total_episodes > 0 else 0.0

    merged = {
        "task_results": merged_task_results,
        "overall_results": {
            "success_rate": overall_sr,
            "total_episodes": total_episodes,
            "total_successes": total_successes,
        },
    }
    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge partial task_comp_l1 summaries")
    parser.add_argument("--summary_dir", type=str, required=True,
                        help="Directory containing partial JSON files")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Seed values to merge (default: 0 1 2)")
    parser.add_argument("--num_parts", type=int, default=2,
                        help="Number of parts per seed (default: 2)")
    args = parser.parse_args()

    for seed in args.seeds:
        print(f"\n--- Merging seed {seed} ---")
        merged = merge_seed(args.summary_dir, seed, args.num_parts)

        out_path = os.path.join(args.summary_dir, f"tinyvla_seed{seed}_merged.json")
        with open(out_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"  Saved: {out_path}")

        # Print summary
        overall = merged["overall_results"]
        print(f"  Tasks: {len(merged['task_results'])}")
        print(f"  Episodes: {overall['total_episodes']}")
        print(f"  Successes: {overall['total_successes']}")
        print(f"  Success rate: {overall['success_rate']:.1%}")

        for task_name, r in merged["task_results"].items():
            sr = r["success_rate"]
            eps = r["episodes"]
            print(f"    {task_name}: {sr:.1%} ({int(sr * eps)}/{eps})")


if __name__ == "__main__":
    main()
