from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.models import clear_gpu
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate


def ablate_kl(config_path: str, kl_values: list[float] | None = None, fork_updates: int | None = None):
    """Run short continuation forks from midpoint across different KL penalty coefficients.

    Isolates the trade-off between reward maximization, policy drift, entropy collapse,
    and qualitative response quality to identify evidence for or against reward overoptimization.
    """
    cfg = load_yaml(config_path)
    betas = kl_values or [float(k) for k in cfg.get("kl_values", [0.0, 0.10, 0.20])]
    updates = int(fork_updates if fork_updates is not None else cfg.get("fork_updates", 8))
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== Starting PPO KL Penalty (Reward Overoptimization) Study ===")
    print(f"KL beta values: {betas}")
    print(f"Fork update budget: {updates} updates")

    summaries = []
    generations_by_beta = {}

    for beta_kl in betas:
        run_name = f"kl_{beta_kl:g}"
        out_dir = repo_path(f"outputs/task2_ppo/{run_name}")

        print(f"\n--- Training PPO fork with kl_beta={beta_kl} ({updates} updates) ---")
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
        eval_summary["updates"] = updates
        summaries.append(eval_summary)
        clear_gpu()

        # Load generated samples for cross-condition qualitative comparison
        gen_file = repo_path(eval_summary["generations_file"])
        if gen_file.exists():
            generations_by_beta[beta_kl] = read_jsonl(gen_file)

    # Save quantitative summary
    summary_file = results_dir / "kl_ablation_summary.json"
    summary_file.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save tabular summary CSV
    df_rows = []
    for s in summaries:
        df_rows.append({
            "kl_beta": s["kl_beta"],
            "updates": s["updates"],
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

    # Perform qualitative comparison for prompts across different beta values
    if len(generations_by_beta) >= 2:
        qualitative_comparisons = []
        base_beta = betas[0]
        base_samples = generations_by_beta.get(base_beta, [])
        for i, base_item in enumerate(base_samples[:15]):  # Compare first 15 prompts
            entry = {
                "prompt": base_item.get("prompt"),
                "variants": {},
            }
            for b in betas:
                samples = generations_by_beta.get(b, [])
                if i < len(samples):
                    entry["variants"][f"beta_{b:g}"] = {
                        "response": samples[i].get("generated"),
                        "reward_score": samples[i].get("reward_score"),
                        "tokens": samples[i].get("tokens"),
                    }
            qualitative_comparisons.append(entry)

        qual_file = results_dir / "kl_qualitative_comparisons.json"
        qual_file.write_text(json.dumps(qualitative_comparisons, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved qualitative cross-condition comparisons to {qual_file}")

    print("\n=== KL Ablation Study Summary ===")
    print(df.to_string(index=False))
    print(f"\nSaved summary logs to {summary_file} and {csv_file}")
    return summaries


def main():
    ap = argparse.ArgumentParser(description="PPO KL penalty (reward overoptimization) ablation study.")
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--kl-values", type=float, nargs="+", help="KL beta values to test.")
    ap.add_argument("--updates", type=int, help="Continuation updates per condition.")
    args = ap.parse_args()

    ablate_kl(args.config, args.kl_values, args.updates)


if __name__ == "__main__":
    main()
