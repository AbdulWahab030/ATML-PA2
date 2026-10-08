from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import torch

from common.data import load_yaml, repo_path
from common.models import clear_gpu
from task1_dpo.evaluate import evaluate
from task1_dpo.train import run_training


def ablate_beta(config_path: str, betas: list[float] | None = None, max_examples: int | None = None):
    """Run short-run DPO ablations across different regularization values beta.

    Trains fresh LoRA adapters for each beta from the base policy initialization
    using a fixed short training budget, and evaluates them under the common protocol.
    """
    cfg = load_yaml(config_path)
    beta_list = betas or [float(b) for b in cfg.get("betas", [0.03, 0.10, 0.30])]
    num_examples = int(max_examples or cfg.get("short_ablation_examples", 600))
    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting DPO Beta Ablation Study ===")
    print(f"Beta values to test: {beta_list}")
    print(f"Examples per condition: {num_examples}")

    summaries = []
    for beta in beta_list:
        run_name = f"beta_{beta:g}"
        out_adapter = repo_path(f"outputs/task1_dpo/{run_name}")
        print(f"\n--- Training DPO with beta={beta} (budget={num_examples} examples) ---")

        run_training(
            config_path=config_path,
            run_name=run_name,
            dataset_path=cfg["paths"]["dpo_standard_train"],
            output_path=str(out_adapter),
            beta=beta,
            max_examples=num_examples,
        )
        clear_gpu()

        print(f"--- Evaluating DPO adapter for beta={beta} ---")
        summary = evaluate(
            config_path=config_path,
            adapter=str(out_adapter),
            name=run_name,
            dataset_path=cfg["paths"]["dpo_standard_eval"],
        )
        summary["ablation_beta"] = beta
        summary["train_examples"] = num_examples
        summaries.append(summary)
        clear_gpu()

    summary_file = results_dir / "beta_ablation_summary.json"
    summary_file.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    # Also save tabular summary for easy plotting and reporting
    df_rows = []
    for s in summaries:
        df_rows.append({
            "beta": s["ablation_beta"],
            "train_examples": s["train_examples"],
            "heldout_dpo_loss": s.get("heldout_dpo_loss"),
            "heldout_preference_accuracy": s.get("heldout_preference_accuracy"),
            "sampled_policy_reference_kl_per_token": s.get("sampled_policy_reference_kl_per_token"),
            "reward_score_mean": s.get("generated_reward_model_score_mean"),
            "reward_score_std": s.get("generated_reward_model_score_std"),
            "response_tokens_mean": s.get("generated_response_tokens_mean"),
            "response_words_mean": s.get("generated_response_words_mean"),
        })
    df = pd.DataFrame(df_rows)
    csv_file = results_dir / "beta_ablation_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n=== Beta Ablation Complete ===")
    print(df.to_string(index=False))
    print(f"Saved aggregated summaries to {summary_file} and {csv_file}")
    return summaries


def main():
    ap = argparse.ArgumentParser(description="Run DPO beta regularization strength ablation study.")
    ap.add_argument("--config", default="configs/dpo.yaml", help="Path to DPO config file.")
    ap.add_argument("--betas", type=float, nargs="+", help="Beta values to sweep over.")
    ap.add_argument("--max-examples", type=int, help="Training budget per condition (default from config).")
    args = ap.parse_args()

    ablate_beta(args.config, args.betas, args.max_examples)


if __name__ == "__main__":
    main()
