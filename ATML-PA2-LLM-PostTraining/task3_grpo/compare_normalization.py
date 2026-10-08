"""task3_grpo/compare_normalization.py — Compare canonical GRPO vs. Dr. GRPO sequence normalization.

Entry-point command (run from the repository root)::

    python -m task3_grpo.compare_normalization --config configs/grpo.yaml

Assignment requirements (Task 3 Step 3):
----------------------------------------
* Run matched short continuations (8 updates) from the identical midpoint checkpoint:
    1. Canonical GRPO: sequence normalization factor (1 / |y_k|).
    2. Dr. GRPO: constant sequence normalization factor (1 / L_max).
* Keep prompts, generation settings, reward model, β, and ε identical.
* Evaluate both checkpoints on held-out evaluation prompts.
* Compare:
    - Held-out reward score
    - Held-out sampled KL divergence
    - Generated response length (mean, std, IQR)
    - Length-conditioned gradient statistics
* Save results to `results/task3_grpo/normalization_comparison.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.data import load_yaml, repo_path
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate


def analyze_length_conditioned_statistics(
    canonical_traj_path: Path,
    dr_traj_path: Path,
    max_completion_length: int,
) -> dict:
    """Analyze gradient weighting and response length shifts between the two normalizations."""
    c_data = json.loads(canonical_traj_path.read_text(encoding="utf-8"))
    d_data = json.loads(dr_traj_path.read_text(encoding="utf-8"))

    c_steps = c_data["trajectory"]
    d_steps = d_data["trajectory"]

    c_lengths = [s["mean_response_length"] for s in c_steps]
    d_lengths = [s["mean_response_length"] for s in d_steps]

    # Effective sequence weighting factor across steps:
    # Under canonical GRPO: token weight = 1 / L
    # Under Dr. GRPO: token weight = 1 / L_max
    # Ratio = L_max / L (quantifies amplification of short sequences relative to L_max)
    c_amplification = [max_completion_length / max(L, 1.0) for L in c_lengths]
    d_amplification = [1.0 for _ in d_lengths]

    return {
        "max_completion_length": max_completion_length,
        "canonical_trajectory_mean_length": round(float(np.mean(c_lengths)), 2),
        "dr_grpo_trajectory_mean_length": round(float(np.mean(d_lengths)), 2),
        "trajectory_length_difference_tokens": round(float(np.mean(d_lengths) - np.mean(c_lengths)), 2),
        "canonical_average_short_sequence_amplification": round(float(np.mean(c_amplification)), 3),
        "dr_grpo_sequence_amplification": 1.0,
        "theoretical_bias_description": (
            "Canonical GRPO scales each token gradient by 1/|y_k|, assigning proportionally larger "
            "gradient updates to short completions (amplification = L_max / |y_k|). Dr. GRPO uses "
            "a constant 1/L_max scaling, removing the penalty against longer, thorough completions."
        ),
    }


def main():
    ap = argparse.ArgumentParser(description="Run and compare canonical GRPO vs. Dr. GRPO normalization.")
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--skip-train", action="store_true", help="Skip training if forks were already run")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    fork_updates = int(cfg.get("fork_updates", 8))
    max_comp_len = int(cfg.get("max_completion_length", 512))

    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    canonical_out = "outputs/task3_grpo/canonical_norm"
    dr_out = "outputs/task3_grpo/dr_grpo_norm"

    if not args.skip_train:
        print("=" * 80)
        print(f"Step 1/2: Running Canonical GRPO Fork (loss_type='grpo', {fork_updates} updates)")
        print("=" * 80)
        run_grpo(
            config_path=args.config,
            output=canonical_out,
            updates=fork_updates,
            loss_type="grpo",
            run_name="canonical_norm",
        )

        print("\n" + "=" * 80)
        print(f"Step 2/2: Running Dr. GRPO Fork (loss_type='dr_grpo', {fork_updates} updates)")
        print("=" * 80)
        run_grpo(
            config_path=args.config,
            output=dr_out,
            updates=fork_updates,
            loss_type="dr_grpo",
            run_name="dr_grpo_norm",
        )

    # Evaluate both checkpoints on held-out evaluation prompts:
    print("\n" + "=" * 80)
    print("Evaluating Canonical GRPO on held-out prompts...")
    print("=" * 80)
    c_eval = evaluate(
        config_path=args.config,
        adapter=canonical_out,
        name="canonical_norm",
    )

    print("\n" + "=" * 80)
    print("Evaluating Dr. GRPO on held-out prompts...")
    print("=" * 80)
    d_eval = evaluate(
        config_path=args.config,
        adapter=dr_out,
        name="dr_grpo_norm",
    )

    # Analyze gradient statistics:
    c_traj_file = results_dir / "canonical_norm_trajectory.json"
    d_traj_file = results_dir / "dr_grpo_norm_trajectory.json"

    length_stats = analyze_length_conditioned_statistics(
        canonical_traj_path=c_traj_file,
        dr_traj_path=d_traj_file,
        max_completion_length=max_comp_len,
    )

    comparison = {
        "fork_updates": fork_updates,
        "canonical_grpo": {
            "adapter": canonical_out,
            "reward_score_mean": c_eval["reward_score_mean"],
            "reward_score_std": c_eval["reward_score_std"],
            "sampled_policy_reference_kl_per_token": c_eval["sampled_policy_reference_kl_per_token"],
            "response_length_tokens_mean": c_eval["response_length_tokens_mean"],
            "response_length_tokens_iqr": c_eval["response_length_tokens_iqr"],
        },
        "dr_grpo": {
            "adapter": dr_out,
            "reward_score_mean": d_eval["reward_score_mean"],
            "reward_score_std": d_eval["reward_score_std"],
            "sampled_policy_reference_kl_per_token": d_eval["sampled_policy_reference_kl_per_token"],
            "response_length_tokens_mean": d_eval["response_length_tokens_mean"],
            "response_length_tokens_iqr": d_eval["response_length_tokens_iqr"],
        },
        "deltas_dr_minus_canonical": {
            "reward_delta": round(d_eval["reward_score_mean"] - c_eval["reward_score_mean"], 4),
            "kl_delta": round(
                d_eval["sampled_policy_reference_kl_per_token"]
                - c_eval["sampled_policy_reference_kl_per_token"],
                5,
            ),
            "length_delta_tokens": round(
                d_eval["response_length_tokens_mean"] - c_eval["response_length_tokens_mean"], 2
            ),
        },
        "length_conditioned_gradient_analysis": length_stats,
    }

    comp_file = results_dir / "normalization_comparison.json"
    comp_file.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("GRPO Normalization Comparison Summary:")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Canonical GRPO':<16} | {'Dr. GRPO':<16} | {'Delta (Dr - Can)':<16}")
    print("-" * 80)
    print(
        f"{'Held-out Reward':<30} | {c_eval['reward_score_mean']:>14.4f} | {d_eval['reward_score_mean']:>14.4f} | "
        f"{comparison['deltas_dr_minus_canonical']['reward_delta']:>+14.4f}"
    )
    print(
        f"{'Sampled KL Drift':<30} | {c_eval['sampled_policy_reference_kl_per_token']:>14.5f} | "
        f"{d_eval['sampled_policy_reference_kl_per_token']:>14.5f} | "
        f"{comparison['deltas_dr_minus_canonical']['kl_delta']:>+14.5f}"
    )
    print(
        f"{'Response Length (Tokens)':<30} | {c_eval['response_length_tokens_mean']:>14.1f} | "
        f"{d_eval['response_length_tokens_mean']:>14.1f} | "
        f"{comparison['deltas_dr_minus_canonical']['length_delta_tokens']:>+14.1f}"
    )
    print("=" * 80)
    print(f"Results saved to: {comp_file}")


if __name__ == "__main__":
    main()
