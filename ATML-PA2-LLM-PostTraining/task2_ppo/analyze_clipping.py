"""task2_ppo/analyze_clipping.py — PPO clipping study: cached geometry + matched forks.

Entry-point command (run from the repository root)::

    python -m task2_ppo.analyze_clipping --config configs/ppo.yaml

Runs two complementary analyses:

**Part A – Offline cached-rollout geometry** (no new training required):
  Loads ``cached/ppo_rollout.pt``, derives or reads per-token advantages,
  then measures for ε ∈ {0.05, 0.20, 0.50}:
    * clip_fraction          — tokens with ρ outside [1−ε, 1+ε].
    * affected_token_fraction — tokens where clipping actively binds.
    * surrogate_objective_mean — mean L_clip before optimisation.

**Part B – Online matched short forks** (requires GPU):
  Continues three independent PPO forks (one per ε) from the midpoint
  checkpoint for ``fork_updates`` steps each.  Evaluates each fork and
  reports: reward, KL, response length, training grad-norm stability,
  and mean clip fraction from the training trajectory.

Results are saved to ``results/task2_ppo/clipping_study_summary.json``
and ``clipping_study_summary.csv``.

Cross-platform notes
--------------------------
``_detect_device`` selects CUDA → MPS → CPU.  ``_inference_autocast``
wraps reward-model calls.  ``clear_gpu`` handles cross-platform cache
clearing.
"""

from __future__ import annotations

import argparse
import contextlib
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
from task2_ppo.continue_train import run_ppo, _detect_device, _inference_autocast
from task2_ppo.evaluate import evaluate
from task2_ppo.ppo import compute_gae, normalize_advantages, shaped_rewards


# ---------------------------------------------------------------------------
# Cached-rollout loader
# ---------------------------------------------------------------------------

def load_cached_rollouts(path) -> list[dict]:
    """Load and normalise the supplied cached PPO rollout batch.

    Accepts both the original field names (``old_policy_logprobs``,
    ``reference_logprobs``) and the simplified names (``old_logprobs``,
    ``ref_logprobs``) used in later releases.

    Args:
        path: Repo-relative or absolute path to the ``.pt`` cache file.

    Returns:
        A list of rollout-row dicts with normalised field names.

    Raises:
        ValueError: If the cache is empty or has an unexpected schema.
    """
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache.")

    normalised: list[dict] = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalised.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalised[0]):
        raise ValueError(
            f"Unexpected PPO cache schema; need at least {sorted(required)}; "
            f"got {sorted(normalised[0])}."
        )
    return normalised


# ---------------------------------------------------------------------------
# Part A – cached rollout geometry
# ---------------------------------------------------------------------------

def analyze_cached_clipping(
    cfg: dict,
    rows: list[dict],
    epsilons: list[float],
    device: torch.device,
) -> list[dict]:
    """Measure clipping geometry on the supplied cached rollouts.

    For each ε value, computes the clip fraction, affected-token fraction,
    and mean surrogate objective directly from the cached importance ratios.
    If advantages are pre-stored in the cache they are used as-is; otherwise
    they are derived from the midpoint policy/value/reward models.

    Args:
        cfg:      Merged PPO config dict.
        rows:     Cached rollout rows from :func:`load_cached_rollouts`.
        epsilons: List of clipping thresholds to evaluate.
        device:   Active compute device.

    Returns:
        List of per-ε metric dicts.
    """
    print("\n--- Part A: Offline Cached Rollout Clipping Analysis ---")

    first_row = rows[0]
    has_advantages = "advantages" in first_row

    if has_advantages:
        print("Using pre-computed advantages from cached rollouts.")
        all_old_logp = torch.cat([r["old_logprobs"].to(device) for r in rows], dim=0)
        all_advantages = torch.cat([r["advantages"].to(device) for r in rows], dim=0)
        all_mask = torch.cat(
            [r.get("mask", torch.ones_like(r["old_logprobs"])).to(device) for r in rows], dim=0
        )
        all_new_logp = torch.cat(
            [r.get("new_logprobs", r["old_logprobs"]).to(device) for r in rows], dim=0
        )
    else:
        print("Deriving advantages from midpoint policy/value/reward models…")
        tokenizer = load_tokenizer(cfg["base_model"])
        policy = load_policy(
            cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=False
        ).to(device)
        value_model = load_value_model(
            cfg, cfg["paths"]["ppo_midpoint_value"], train_mode="frozen"
        ).to(device)
        reward_model, reward_tokenizer = load_reward_model(cfg)

        old_logp_list, new_logp_list, adv_list, mask_list = [], [], [], []

        for r in rows:
            old_lp = r["old_logprobs"].to(device)
            ref_lp = r["ref_logprobs"].to(device)
            resp = str(r["response"])
            prompt_idx = r.get("source_index", 0)

            resp_enc = tokenizer(resp, return_tensors="pt", add_special_tokens=False)
            resp_ids = resp_enc["input_ids"].to(device)
            mask = torch.ones_like(resp_ids, dtype=torch.float32, device=device)

            if "rewards" in r:
                task_rew = torch.as_tensor([float(r["rewards"])], device=device)
            else:
                with _inference_autocast(device):
                    task_rew = score_reward_pairs(
                        reward_model,
                        reward_tokenizer,
                        [[{"role": "user", "content": f"Prompt {prompt_idx}"}]],
                        [resp],
                    ).to(device)

            vals = r["values"].to(device) if "values" in r else torch.zeros_like(
                old_lp, dtype=torch.float32
            )

            shaped_rew = shaped_rewards(
                task_rew, old_lp, ref_lp, mask, float(cfg.get("kl_beta", 0.10))
            )
            adv, _ = compute_gae(
                shaped_rew, vals, mask,
                gamma=float(cfg.get("gamma", 1.0)),
                lam=float(cfg.get("gae_lambda", 0.95)),
            )
            norm_adv = normalize_advantages(adv, mask)

            old_logp_list.append(old_lp.squeeze(0) if old_lp.dim() > 1 else old_lp)
            new_logp_list.append(old_lp.squeeze(0) if old_lp.dim() > 1 else old_lp)
            adv_list.append(norm_adv.squeeze(0) if norm_adv.dim() > 1 else norm_adv)
            mask_list.append(mask.squeeze(0) if mask.dim() > 1 else mask)

        all_old_logp = torch.cat(old_logp_list, dim=-1)
        all_new_logp = torch.cat(new_logp_list, dim=-1)
        all_advantages = torch.cat(adv_list, dim=-1)
        all_mask = torch.cat(mask_list, dim=-1)

        clear_gpu(policy, value_model, reward_model)

    diff = torch.nan_to_num(all_new_logp - all_old_logp, nan=0.0, posinf=20.0, neginf=-20.0)
    ratio = torch.exp(torch.clamp(diff, min=-20.0, max=20.0))
    ratio = torch.nan_to_num(ratio, nan=1.0)
    cached_metrics: list[dict] = []

    for eps in epsilons:
        surr1 = ratio * all_advantages
        surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * all_advantages
        objective = torch.minimum(surr1, surr2)

        # Tokens with ρ outside [1−ε, 1+ε].
        out_of_bounds = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
        clip_fraction = float(masked_mean(out_of_bounds, all_mask).item())

        # Tokens where clipping actively binds (surr2 < surr1).
        clipping_binds = (surr2 < surr1).float()
        affected_fraction = float(masked_mean(clipping_binds, all_mask).item())

        mean_obj = float(masked_mean(objective, all_mask).item())
        cached_metrics.append({
            "epsilon": eps,
            "clip_fraction": clip_fraction,
            "affected_token_fraction": affected_fraction,
            "surrogate_objective_mean": mean_obj,
        })
        print(
            f"  ε={eps:.2f}  clip_fraction={clip_fraction:.4f}  "
            f"affected_tokens={affected_fraction:.4f}  mean_obj={mean_obj:.4f}"
        )

    return cached_metrics


# ---------------------------------------------------------------------------
# Part B – matched online short forks
# ---------------------------------------------------------------------------

def run_clipping_forks(
    config_path: str,
    epsilons: list[float],
    fork_updates: int,
) -> list[dict]:
    """Run matched short PPO forks from the midpoint for each ε value.

    Each fork starts from the identical supplied midpoint checkpoint so
    results are directly comparable across ε values.

    Args:
        config_path:  Path to ``configs/ppo.yaml``.
        epsilons:     List of clipping thresholds to test.
        fork_updates: Number of continuation update steps per fork.

    Returns:
        List of per-fork summary dicts with reward, KL, length, and
        training-stability statistics.
    """
    print(f"\n--- Part B: Online Matched Continuation Forks ({fork_updates} updates each) ---")
    cfg = load_yaml(config_path)
    fork_summaries: list[dict] = []

    for eps in epsilons:
        run_name = f"clip_{eps:g}"
        out_dir = repo_path(f"outputs/task2_ppo/{run_name}")

        print(f"\nTraining PPO fork with ε={eps}…")
        train_res = run_ppo(
            config_path=config_path,
            output=str(out_dir),
            updates=fork_updates,
            clip_epsilon=eps,
            kl_beta=float(cfg.get("kl_beta", 0.10)),
            run_name=run_name,
        )
        clear_gpu()

        print(f"Evaluating PPO fork adapter: {run_name}…")
        eval_summary = evaluate(
            config_path=config_path,
            adapter=str(out_dir),
            name=run_name,
        )
        clear_gpu()

        # Extract per-step training statistics from the trajectory log.
        traj_file = repo_path(train_res["trajectory_file"])
        grad_norm_std = loss_std = mean_clip_frac = None
        if traj_file.exists():
            traj = json.loads(traj_file.read_text(encoding="utf-8"))
            pi_grad_norms = [float(s.get("policy_grad_norm", 0.0)) for s in traj]
            pi_losses = [float(s.get("policy_loss", 0.0)) for s in traj]
            clip_fractions = [float(s.get("clip_fraction", 0.0)) for s in traj]
            grad_norm_std = float(np.std(pi_grad_norms))
            loss_std = float(np.std(pi_losses))
            mean_clip_frac = float(np.mean(clip_fractions))

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry-point for the PPO clipping study."""
    ap = argparse.ArgumentParser(
        description="PPO clipping study: cached rollout geometry + matched short forks."
    )
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--epsilons", type=float, nargs="+",
                    help="ε values to sweep (default from config).")
    ap.add_argument("--skip-forks", action="store_true",
                    help="Only run Part A (cached rollout analysis); skip fork training.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = _detect_device()
    print(f"[device] Using: {device}")

    epsilons = args.epsilons or [float(e) for e in cfg.get("clip_values", [0.05, 0.20, 0.50])]
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # Part A — cached rollout analysis.
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    cached_metrics = analyze_cached_clipping(cfg, rows, epsilons, device)

    # Part B — matched online forks.
    fork_summaries: list[dict] = []
    if not args.skip_forks:
        fork_updates = int(cfg.get("fork_updates", 8))
        fork_summaries = run_clipping_forks(args.config, epsilons, fork_updates)

    combined = {
        "cached_rollout_analysis": cached_metrics,
        "online_fork_analysis": fork_summaries,
    }
    summary_file = results_dir / "clipping_study_summary.json"
    summary_file.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")

    if fork_summaries:
        cached_map = {m["epsilon"]: m for m in cached_metrics}
        merged_rows = []
        for fork in fork_summaries:
            eps = fork["epsilon"]
            c = cached_map.get(eps, {})
            merged_rows.append({
                "epsilon": eps,
                "cached_clip_fraction": c.get("clip_fraction"),
                "cached_affected_fraction": c.get("affected_token_fraction"),
                "heldout_reward_mean": fork.get("heldout_reward_mean"),
                "heldout_kl_per_token": fork.get("heldout_kl_per_token"),
                "heldout_response_tokens_mean": fork.get("heldout_response_tokens_mean"),
                "training_grad_norm_std": fork.get("training_grad_norm_std"),
            })
        df = pd.DataFrame(merged_rows)
        csv_file = results_dir / "clipping_study_summary.csv"
        df.to_csv(csv_file, index=False)
        print("\n=== Clipping Study Summary Table ===")
        print(df.to_string(index=False))

    print(f"\nSaved clipping study results → {summary_file}")


if __name__ == "__main__":
    main()
