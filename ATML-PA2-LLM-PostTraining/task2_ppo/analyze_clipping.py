from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, prompt_messages, repo_path
from common.generation import response_token_logprobs, score_reward_pairs
from common.metrics import masked_mean
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
)
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate
from task2_ppo.ppo import compute_gae, normalize_advantages, shaped_rewards


def load_cached_rollouts(path):
    """Load and normalize the supplied cached PPO rollout batch."""
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def analyze_cached_clipping(cfg: dict, rows: list[dict], epsilons: list[float], device):
    """Measure immediate clipping geometry and affected-token fractions on cached rollouts."""
    print("\n--- Part A: Offline Cached Rollout Clipping Analysis ---")
    cached_metrics = []

    # Check if advantages are already stored in the cache
    first_row = rows[0]
    has_advantages = "advantages" in first_row

    if has_advantages:
        print("Using precomputed advantages from cached rollouts.")
        all_old_logp = torch.cat([r["old_logprobs"].to(device) for r in rows], dim=0)
        all_advantages = torch.cat([r["advantages"].to(device) for r in rows], dim=0)
        all_mask = torch.cat([r.get("mask", torch.ones_like(r["old_logprobs"])).to(device) for r in rows], dim=0)
        # If new_logprobs exists, use it; otherwise use old_logprobs as baseline snapshot
        all_new_logp = torch.cat([r.get("new_logprobs", r["old_logprobs"]).to(device) for r in rows], dim=0)
    else:
        print("Deriving advantages from cached rollouts using midpoint policy and value models...")
        # Prepare models to evaluate cached batch
        tokenizer = load_tokenizer(cfg["base_model"])
        policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=False).to(device)
        value_model = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], train_mode="frozen").to(device)
        reward_model, reward_tokenizer = load_reward_model(cfg)

        old_logp_list, new_logp_list, adv_list, mask_list = [], [], [], []

        for r in rows:
            old_lp = r["old_logprobs"].to(device)
            ref_lp = r["ref_logprobs"].to(device)
            resp = str(r["response"])
            prompt_idx = r.get("source_index", 0)

            # Reconstruct tokens
            resp_enc = tokenizer(resp, return_tensors="pt", add_special_tokens=False)
            resp_ids = resp_enc["input_ids"].to(device)
            mask = torch.ones_like(resp_ids, dtype=torch.float32, device=device)

            if "rewards" in r:
                task_rew = torch.as_tensor([float(r["rewards"])], device=device)
            else:
                prompts = [[{"role": "user", "content": f"Prompt {prompt_idx}"}]]
                task_rew = score_reward_pairs(reward_model, reward_tokenizer, prompts, [resp]).to(device)

            if "values" in r:
                vals = r["values"].to(device)
            else:
                vals = torch.zeros_like(resp_ids, dtype=torch.float32)

            shaped_rew = shaped_rewards(task_rew, old_lp, ref_lp, mask, float(cfg.get("kl_beta", 0.10)))
            adv, _ = compute_gae(shaped_rew, vals, mask, gamma=float(cfg.get("gamma", 1.0)), lam=float(cfg.get("gae_lambda", 0.95)))
            norm_adv = normalize_advantages(adv, mask)

            old_logp_list.append(old_lp)
            new_logp_list.append(old_lp)  # At snapshot midpoint, ratio = 1.0 unless updated
            adv_list.append(norm_adv)
            mask_list.append(mask)

        all_old_logp = torch.cat(old_logp_list, dim=-1)
        all_new_logp = torch.cat(new_logp_list, dim=-1)
        all_advantages = torch.cat(adv_list, dim=-1)
        all_mask = torch.cat(mask_list, dim=-1)

        clear_gpu(policy, value_model, reward_model)

    ratio = torch.exp(torch.clamp(all_new_logp - all_old_logp, min=-20.0, max=20.0))

    for eps in epsilons:
        surr1 = ratio * all_advantages
        surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * all_advantages
        objective = torch.minimum(surr1, surr2)

        # Fraction of valid tokens outside [1 - eps, 1 + eps]
        out_of_bounds = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
        clip_fraction = float(masked_mean(out_of_bounds, all_mask).item())

        # Affected tokens: tokens where clipping actively binds (surr2 < surr1)
        clipping_binds = (surr2 < surr1).float()
        affected_fraction = float(masked_mean(clipping_binds, all_mask).item())

        mean_obj = float(masked_mean(objective, all_mask).item())

        cached_metrics.append({
            "epsilon": eps,
            "clip_fraction": clip_fraction,
            "affected_token_fraction": affected_fraction,
            "surrogate_objective_mean": mean_obj,
        })
        print(f"epsilon={eps:.2f} -> clip_fraction={clip_fraction:.4f}, affected_tokens={affected_fraction:.4f}, mean_obj={mean_obj:.4f}")

    return cached_metrics


def run_clipping_forks(config_path: str, epsilons: list[float], fork_updates: int):
    """Run matched short continuation forks from midpoint for each epsilon."""
    print(f"\n--- Part B: Online Matched Continuation Forks ({fork_updates} updates) ---")
    cfg = load_yaml(config_path)
    fork_summaries = []

    for eps in epsilons:
        run_name = f"clip_{eps:g}"
        out_dir = repo_path(f"outputs/task2_ppo/{run_name}")

        print(f"\nTraining PPO fork with epsilon={eps}...")
        train_res = run_ppo(
            config_path=config_path,
            output=str(out_dir),
            updates=fork_updates,
            clip_epsilon=eps,
            kl_beta=float(cfg.get("kl_beta", 0.10)),
            run_name=run_name,
        )
        clear_gpu()

        print(f"Evaluating PPO fork adapter: {run_name}...")
        eval_summary = evaluate(
            config_path=config_path,
            adapter=str(out_dir),
            name=run_name,
        )
        clear_gpu()

        # Compute optimization stability statistics from the trajectory log
        traj_file = repo_path(train_res["trajectory_file"])
        if traj_file.exists():
            traj = json.loads(traj_file.read_text(encoding="utf-8"))
            pi_grad_norms = [float(step.get("policy_grad_norm", 0.0)) for step in traj]
            pi_losses = [float(step.get("policy_loss", 0.0)) for step in traj]
            clip_fractions = [float(step.get("clip_fraction", 0.0)) for step in traj]

            grad_norm_std = float(np.std(pi_grad_norms))
            loss_std = float(np.std(pi_losses))
            mean_clip_frac = float(np.mean(clip_fractions))
        else:
            grad_norm_std, loss_std, mean_clip_frac = None, None, None

        fork_summaries.append({
            "epsilon": eps,
            "heldout_reward_mean": eval_summary.get("reward_score_mean"),
            "heldout_reward_std": eval_summary.get("reward_score_std"),
            "heldout_kl_per_token": eval_summary.get("sampled_policy_reference_kl_per_token"),
            "heldout_response_tokens_mean": eval_summary.get("response_tokens_mean"),
            "training_grad_norm_std": grad_norm_std,
            "training_loss_std": loss_std,
            "training_mean_clip_fraction": mean_clip_frac,
        })

    return fork_summaries


def main():
    ap = argparse.ArgumentParser(description="PPO clipping study: cached rollout geometry and matched short forks.")
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--epsilons", type=float, nargs="+", help="Epsilon values to test.")
    ap.add_argument("--skip-forks", action="store_true", help="Only run cached rollout analysis.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    epsilons = args.epsilons or [float(e) for e in cfg.get("clip_values", [0.05, 0.20, 0.50])]
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Part A: Cached rollouts
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    cached_metrics = analyze_cached_clipping(cfg, rows, epsilons, device)

    # 2. Part B: Matched forks
    fork_summaries = []
    if not args.skip_forks:
        fork_updates = int(cfg.get("fork_updates", 8))
        fork_summaries = run_clipping_forks(args.config, epsilons, fork_updates)

    combined_summary = {
        "cached_rollout_analysis": cached_metrics,
        "online_fork_analysis": fork_summaries,
    }
    summary_file = results_dir / "clipping_study_summary.json"
    summary_file.write_text(json.dumps(combined_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save tabular CSV summary
    if fork_summaries:
        cached_map = {m["epsilon"]: m for m in cached_metrics}
        merged_rows = []
        for fork in fork_summaries:
            eps = fork["epsilon"]
            c_info = cached_map.get(eps, {})
            merged_rows.append({
                "epsilon": eps,
                "cached_clip_fraction": c_info.get("clip_fraction"),
                "cached_affected_fraction": c_info.get("affected_token_fraction"),
                "heldout_reward_mean": fork.get("heldout_reward_mean"),
                "heldout_kl_per_token": fork.get("heldout_kl_per_token"),
                "heldout_response_tokens_mean": fork.get("heldout_response_tokens_mean"),
                "grad_norm_std": fork.get("training_grad_norm_std"),
            })
        df = pd.DataFrame(merged_rows)
        csv_file = results_dir / "clipping_study_summary.csv"
        df.to_csv(csv_file, index=False)
        print("\n=== Clipping Study Summary Table ===")
        print(df.to_string(index=False))

    print(f"\nSaved clipping study results to {summary_file}")


if __name__ == "__main__":
    main()
