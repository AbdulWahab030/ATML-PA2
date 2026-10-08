"""task2_ppo/ablate_kl.py — KL penalty ablation study (reward over-optimisation).

Entry-point command (run from the repository root)::

    python -m task2_ppo.ablate_kl --config configs/ppo.yaml

Trains three independent short PPO forks from the supplied midpoint
checkpoint, one for each KL penalty coefficient β_KL ∈ {0, 0.10, 0.20}.
All other hyperparameters (clip ε, learning rates, update count) are held
fixed to isolate the effect of the KL penalty on:

  * Reward maximisation (reward-model scores).
  * Policy drift from the reference (sampled KL/token).
  * Policy entropy and response length.
  * Reward over-optimisation signatures (reward ↑ while quality ↓).

Results are saved to:
  ``results/task2_ppo/kl_ablation_summary.json``
  ``results/task2_ppo/kl_ablation_summary.csv``
  ``results/task2_ppo/kl_qualitative_comparisons.json``

Cross-platform notes
--------------------------
``_detect_device`` and ``clear_gpu`` from the continuation module provide
cross-platform CUDA / MPS / CPU handling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.models import clear_gpu
from task2_ppo.continue_train import _detect_device, run_ppo
from task2_ppo.evaluate import evaluate


# ---------------------------------------------------------------------------
# Ablation driver
# ---------------------------------------------------------------------------

def ablate_kl(
    config_path: str,
    kl_values: list[float] | None = None,
    fork_updates: int | None = None,
) -> list[dict]:
    """Run short PPO continuation forks across different KL penalty coefficients.

    Each fork starts from the *identical* supplied midpoint checkpoint so
    results are directly comparable across β_KL values.  The analysis
    probes the reward over-optimisation regime:

    * β_KL = 0   — no regularisation; maximum reward hacking risk.
    * β_KL = 0.10 — default configuration used in standard continuation.
    * β_KL = 0.20 — stronger regularisation; policy stays near reference.

    Args:
        config_path: Path to ``configs/ppo.yaml``.
        kl_values:   Override list of β_KL values to sweep
                     (default from ``kl_values`` key in config).
        fork_updates: Override number of update steps per fork
                      (default from ``fork_updates`` key in config).

    Returns:
        List of per-β_KL summary dicts with evaluation metrics.
    """
    cfg = load_yaml(config_path)
    device = _detect_device()
    print(f"[device] Using: {device}")

    betas = kl_values or [float(k) for k in cfg.get("kl_values", [0.0, 0.10, 0.20])]
    updates = int(fork_updates if fork_updates is not None else cfg.get("fork_updates", 8))
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== Starting PPO KL Penalty (Reward Over-Optimisation) Study ===")
    print(f"β_KL values : {betas}")
    print(f"Fork updates: {updates}")

    summaries: list[dict] = []
    generations_by_beta: dict[float, list[dict]] = {}

    for beta_kl in betas:
        run_name = f"kl_{beta_kl:g}"
        out_dir = repo_path(f"outputs/task2_ppo/{run_name}")

        print(f"\n--- Training PPO fork with β_KL={beta_kl} ({updates} updates) ---")
        run_ppo(
            config_path=config_path,
            output=str(out_dir),
            updates=updates,
            clip_epsilon=float(cfg.get("clip_epsilon", 0.20)),
            kl_beta=beta_kl,
            run_name=run_name,
        )
        clear_gpu()

        print(f"--- Evaluating PPO fork adapter: {run_name} ---")
        eval_summary = evaluate(
            config_path=config_path,
            adapter=str(out_dir),
            name=run_name,
        )
        eval_summary["kl_beta"] = beta_kl
        eval_summary["fork_updates"] = updates
        summaries.append(eval_summary)
        clear_gpu()

        # Cache generated samples for cross-condition qualitative comparison.
        gen_file = repo_path(eval_summary["generations_file"])
        if gen_file.exists():
            generations_by_beta[beta_kl] = read_jsonl(gen_file)

    # Save quantitative summary.
    summary_file = results_dir / "kl_ablation_summary.json"
    summary_file.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save tabular CSV for plotting.
    df_rows = []
    for s in summaries:
        df_rows.append({
            "kl_beta": s["kl_beta"],
            "fork_updates": s["fork_updates"],
            "heldout_reward_mean": s.get("reward_score_mean"),
            "heldout_reward_std": s.get("reward_score_std"),
            "heldout_kl_per_token": s.get("sampled_policy_reference_kl_per_token"),
            "response_tokens_mean": s.get("response_tokens_mean"),
            "response_words_mean": s.get("response_words_mean"),
            "truncated_count": s.get("truncated_count"),
        })
    df = pd.DataFrame(df_rows)
    csv_file = results_dir / "kl_ablation_summary.csv"
    df.to_csv(csv_file, index=False)

    # Qualitative cross-condition comparison (first 15 prompts).
    if len(generations_by_beta) >= 2:
        base_beta = betas[0]
        base_samples = generations_by_beta.get(base_beta, [])
        qualitative: list[dict] = []
        for i, base_item in enumerate(base_samples[:15]):
            entry: dict = {"prompt": base_item.get("prompt"), "variants": {}}
            for b in betas:
                samples = generations_by_beta.get(b, [])
                if i < len(samples):
                    entry["variants"][f"beta_{b:g}"] = {
                        "response": samples[i].get("generated"),
                        "reward_score": samples[i].get("reward_score"),
                        "tokens": samples[i].get("tokens"),
                    }
            qualitative.append(entry)
        qual_file = results_dir / "kl_qualitative_comparisons.json"
        qual_file.write_text(
            json.dumps(qualitative, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Saved qualitative cross-condition comparisons → {qual_file}")

    print("\n=== KL Ablation Study Summary ===")
    print(df.to_string(index=False))
    print(f"\nSaved summary logs → {summary_file} and {csv_file}")
    return summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry-point for the PPO KL penalty ablation study."""
    ap = argparse.ArgumentParser(
        description="PPO KL penalty (reward over-optimisation) ablation study."
    )
    ap.add_argument("--config", default="configs/ppo.yaml",
                    help="Path to the task PPO config.")
    ap.add_argument("--kl-values", type=float, nargs="+",
                    help="β_KL values to sweep (default from config).")
    ap.add_argument("--updates", type=int,
                    help="Number of continuation update steps per fork (default from config).")
    args = ap.parse_args()

    ablate_kl(args.config, args.kl_values, args.updates)


if __name__ == "__main__":
    main()
