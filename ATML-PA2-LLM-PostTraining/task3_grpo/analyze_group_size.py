"""task3_grpo/analyze_group_size.py — Equal-generation group-size diagnostic study.

Entry-point command (run from the repository root)::

    python -m task3_grpo.analyze_group_size --config configs/grpo.yaml

Assignment requirements (Task 3 Step 2):
----------------------------------------
* Using the supplied completion/reward cache (cached/grpo_k_cache.jsonl), regroup
  the same fixed generation budget into K ∈ {2, 4, 8}.
* Report:
    1. Informative-group rate (fraction with σ_r > 0 under numerical tolerance).
    2. Mean within-group reward standard deviation.
    3. Variance of the group-relative signal A_k = (r_k − μ_r) / (σ_r + ε).
    4. The same quantities stratified across at least two prompt-difficulty bins
       (defined via median prompt-level reward: Hard vs Easy).
* Save machine-readable output to `results/task3_grpo/group_size_analysis.json`.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path


def load_k8_cache(path: str | Path) -> dict[str, list[dict]]:
    """Load cached completions and group by source prompt index."""
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)

    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")

    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(
    by_prompt: dict[str, list[dict]],
    k: int,
) -> list[dict]:
    """Partition cached completions into K-sized groups at equal total generation budget.

    Strategy:
    Each prompt has 8 cached completions. For group size K ∈ {2, 4, 8}, each prompt's
    8 completions are partitioned into 8 / K non-overlapping within-prompt groups:
      - K = 8: 1 group of 8 completions per prompt (24 groups * 8 = 192 completions)
      - K = 4: 2 groups of 4 completions per prompt (48 groups * 4 = 192 completions)
      - K = 2: 4 groups of 2 completions per prompt (96 groups * 2 = 192 completions)

    Total generation budget is held exactly fixed at 192 across all conditions.

    Args:
        by_prompt: Mapping from prompt_id -> list of 8 completion dicts.
        k:         Target group size (2, 4, or 8).

    Returns:
        List of group records, each containing `prompt_id`, `completions`, and `rewards`.
    """
    if 8 % k != 0:
        raise ValueError(f"k={k} must divide 8 for exact partitioning")

    groups = []
    num_subgroups = 8 // k

    for pid, completions in by_prompt.items():
        for sub_idx in range(num_subgroups):
            chunk = completions[sub_idx * k : (sub_idx + 1) * k]
            rewards = [float(item["reward"]) for item in chunk]
            groups.append({
                "prompt_id": pid,
                "subgroup_index": sub_idx,
                "k": k,
                "rewards": rewards,
                "completions": chunk,
            })

    return groups


def compute_group_metrics(groups: list[dict], eps: float = 1e-6, zero_std_tol: float = 1e-5) -> dict:
    """Compute informative-group rate, mean std, and advantage variance for a set of groups."""
    if not groups:
        return {
            "num_groups": 0,
            "total_generations": 0,
            "informative_group_rate": 0.0,
            "uninformative_group_rate": 1.0,
            "mean_within_group_reward_std": 0.0,
            "group_relative_signal_variance": 0.0,
        }

    stds = []
    informative_count = 0
    all_advantages = []

    for g in groups:
        r = np.array(g["rewards"], dtype=float)
        mean_r = np.mean(r)
        std_r = np.std(r)  # population std matching 1/K formula

        stds.append(float(std_r))
        is_informative = bool(std_r > zero_std_tol)
        if is_informative:
            informative_count += 1
            adv = (r - mean_r) / (std_r + eps)
        else:
            # When std_r == 0, r - mean_r is identically 0
            adv = np.zeros_like(r)

        all_advantages.extend(adv.tolist())

    n_groups = len(groups)
    inf_rate = informative_count / n_groups
    mean_std = float(np.mean(stds))
    adv_var = float(np.var(all_advantages))

    return {
        "num_groups": n_groups,
        "total_generations": n_groups * len(groups[0]["rewards"]),
        "informative_group_rate": round(inf_rate, 4),
        "uninformative_group_rate": round(1.0 - inf_rate, 4),
        "mean_within_group_reward_std": round(mean_std, 4),
        "group_relative_signal_variance": round(adv_var, 4),
    }


def analyze_group_sizes(cache_path: str | Path, group_sizes: list[int]) -> dict:
    """Run the complete group-size diagnostic across all prompts and difficulty strata."""
    by_prompt = load_k8_cache(cache_path)

    # Compute prompt-level difficulty via mean reward across all 8 cached rollouts:
    prompt_mean_rewards = {
        pid: float(np.mean([item["reward"] for item in comps]))
        for pid, comps in by_prompt.items()
    }
    median_reward = float(np.median(list(prompt_mean_rewards.values())))

    hard_pids = {pid for pid, r in prompt_mean_rewards.items() if r <= median_reward}
    easy_pids = {pid for pid, r in prompt_mean_rewards.items() if r > median_reward}

    results = {
        "total_prompts": len(by_prompt),
        "total_cached_generations": len(by_prompt) * 8,
        "prompt_difficulty_split": {
            "criterion": "median_mean_reward",
            "median_threshold": round(median_reward, 4),
            "num_hard_prompts": len(hard_pids),
            "num_easy_prompts": len(easy_pids),
        },
        "by_k": {},
    }

    print("=" * 80)
    print(f"GRPO Group-Size Diagnostic (Equal Generation Budget = {results['total_cached_generations']})")
    print(f"Difficulty split: Hard (R <= {median_reward:.3f}, N={len(hard_pids)}) vs Easy (R > {median_reward:.3f}, N={len(easy_pids)})")
    print("=" * 80)
    print(f"{'K':<4} | {'Subset':<6} | {'Groups':<6} | {'Generations':<11} | {'Informative %':<13} | {'Mean Std':<10} | {'Adv Var':<10}")
    print("-" * 80)

    for k in group_sizes:
        all_groups = regroup_equal_generation_budget(by_prompt, k)
        hard_groups = [g for g in all_groups if g["prompt_id"] in hard_pids]
        easy_groups = [g for g in all_groups if g["prompt_id"] in easy_pids]

        all_metrics = compute_group_metrics(all_groups)
        hard_metrics = compute_group_metrics(hard_groups)
        easy_metrics = compute_group_metrics(easy_groups)

        results["by_k"][f"K={k}"] = {
            "all": all_metrics,
            "hard": hard_metrics,
            "easy": easy_metrics,
        }

        print(
            f"{k:<4} | {'All':<6} | {all_metrics['num_groups']:<6} | {all_metrics['total_generations']:<11} | "
            f"{all_metrics['informative_group_rate']*100:>11.1f}% | {all_metrics['mean_within_group_reward_std']:>8.4f} | "
            f"{all_metrics['group_relative_signal_variance']:>8.4f}"
        )
        print(
            f"{'':<4} | {'Hard':<6} | {hard_metrics['num_groups']:<6} | {hard_metrics['total_generations']:<11} | "
            f"{hard_metrics['informative_group_rate']*100:>11.1f}% | {hard_metrics['mean_within_group_reward_std']:>8.4f} | "
            f"{hard_metrics['group_relative_signal_variance']:>8.4f}"
        )
        print(
            f"{'':<4} | {'Easy':<6} | {easy_metrics['num_groups']:<6} | {easy_metrics['total_generations']:<11} | "
            f"{easy_metrics['informative_group_rate']*100:>11.1f}% | {easy_metrics['mean_within_group_reward_std']:>8.4f} | "
            f"{easy_metrics['group_relative_signal_variance']:>8.4f}"
        )
        print("-" * 80)

    return results


def main():
    ap = argparse.ArgumentParser(description="Analyze GRPO group sizes under equal total generation budget.")
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cache_path = repo_path(cfg["group_cache"])
    group_sizes = [int(k) for k in cfg.get("group_sizes", [2, 4, 8])]

    results = analyze_group_sizes(cache_path, group_sizes)

    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "group_size_analysis.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved group-size analysis to {out_file}")


if __name__ == "__main__":
    main()
