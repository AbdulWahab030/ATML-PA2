"""task3_grpo/continue_train.py — Standard GRPO continuation from the course midpoint.

Entry-point command (run from the repository root)::

    python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard

This script:
1. Loads the supplied midpoint policy checkpoint (checkpoints/grpo_midpoint_policy).
2. Generates K completions per prompt (critic-free on-policy rollouts).
3. Masks truncated max-length completions when configured.
4. Computes within-prompt group-relative advantages.
5. Applies the clipped GRPO objective (canonical or Dr. GRPO constant length cap).
6. Logs trajectory diagnostics (reward, KL, within-group reward std, uninformative fraction,
   entropy, response length, gradient norm, peak memory, wall-clock time) to
   `results/task3_grpo/<run_name>_trajectory.json`.
7. Saves the trained adapter to `outputs/task3_grpo/<run_name>/`.

Cross-platform device handling
---------------------------------
* Device is dynamically selected: CUDA → MPS → CPU.
* `torch.autocast(device_type=device.type)` with graceful fallback to float32.
* Safe memory clearing and peak memory query across CUDA / MPS / CPU.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import set_seed
from common.metrics import masked_mean, mean_response_length, sample_entropy, sampled_kl
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task3_grpo.grpo import (
    compute_group_statistics,
    group_relative_advantages,
    grpo_policy_loss,
    mask_truncated_sequences,
)


# ---------------------------------------------------------------------------
# Cross-platform device utilities
# ---------------------------------------------------------------------------

def _detect_device() -> torch.device:
    """Return the best available compute device dynamically."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _reset_peak_memory(device: torch.device) -> None:
    """Reset peak-memory tracking if supported by device."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps" and hasattr(torch.mps, "reset_peak_memory_stats"):
        try:
            torch.mps.reset_peak_memory_stats()
        except Exception:
            pass


def get_peak_memory_mb(device: torch.device) -> float:
    """Return peak allocated device memory in megabytes."""
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        try:
            return float(torch.mps.current_allocated_memory() / (1024 * 1024))
        except Exception:
            return 0.0
    return 0.0


@contextlib.contextmanager
def _inference_autocast(device: torch.device):
    """Context manager for mixed-precision inference with cross-platform fallback."""
    if device.type in {"cuda", "mps"}:
        try:
            with torch.autocast(device_type=device.type, enabled=True):
                yield
            return
        except (RuntimeError, TypeError):
            pass
    with contextlib.nullcontext():
        yield


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def prepare_grpo_continuation(config_path: str) -> dict:
    """Load models, tokenizer, prompt pool, and optimizer for GRPO continuation."""
    cfg = load_yaml(config_path)
    set_seed(int(cfg.get("seed", 6304)))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_grpo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    loss_type: str = "grpo",
    run_name: str = "standard",
    clip_epsilon: float | None = None,
    kl_beta: float | None = None,
) -> dict:
    """Run online GRPO continuation updates from the supplied midpoint checkpoint.

    Args:
        config_path:  Path to `configs/grpo.yaml`.
        output:       Override output directory for the saved policy adapter.
        updates:      Override number of continuation update steps.
        loss_type:    `'grpo'` (canonical 1/|y_k|) or `'dr_grpo'` (1/L_max cap).
        run_name:     Identifier used for output files and logs.
        clip_epsilon: Override clipping parameter ε.
        kl_beta:      Override KL coefficient β.

    Returns:
        Dict with execution summary and paths.
    """
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]

    device = _detect_device()
    print(f"[device] Using: {device}")

    policy = bundle["policy"].to(device)
    tokenizer = bundle["tokenizer"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    prompt_rows = bundle["prompt_rows"]
    optimizer = bundle["optimizer"]

    total_updates = int(updates if updates is not None else cfg.get("updates", 20))
    clip_eps = float(clip_epsilon if clip_epsilon is not None else cfg.get("clip_epsilon", 0.20))
    beta_kl = float(kl_beta if kl_beta is not None else cfg.get("kl_beta", 0.10))
    k_generations = int(cfg.get("num_generations", 4))
    prompts_per_update = int(cfg.get("prompts_per_update", 1))
    policy_epochs = int(cfg.get("policy_epochs", 1))
    max_prompt_length = int(cfg.get("max_prompt_length", 256))
    max_completion_length = int(cfg.get("max_completion_length", 512))
    mask_truncated = bool(cfg.get("mask_truncated_completions", True))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    reward_max_length = int(cfg.get("reward_max_length", 1280))
    generation_cfg = cfg.get("generation", {})

    out_path = repo_path(output or cfg.get("output", f"outputs/task3_grpo/{run_name}"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting GRPO Continuation: {run_name} ===")
    print(f"Updates: {total_updates}  K: {k_generations}  loss_type: {loss_type}  clip_eps: {clip_eps}  beta_kl: {beta_kl}")
    print(f"Output adapter: {out_path}")

    trajectory: list[dict] = []
    start_total_time = time.time()
    num_prompts = len(prompt_rows)

    _reset_peak_memory(device)

    for step in range(1, total_updates + 1):
        step_start = time.time()

        # ------------------------------------------------------------------
        # 1. Select prompts and construct K rollouts per prompt
        # ------------------------------------------------------------------
        p_start = ((step - 1) * prompts_per_update) % num_prompts
        p_end = p_start + prompts_per_update
        if p_end <= num_prompts:
            batch_rows = prompt_rows[p_start:p_end]
        else:
            batch_rows = prompt_rows[p_start:] + prompt_rows[: p_end % num_prompts]

        all_prompts = []
        group_id_list = []
        for g_idx, r in enumerate(batch_rows):
            p_msgs = prompt_messages(r)
            for _ in range(k_generations):
                all_prompts.append(p_msgs)
                group_id_list.append(g_idx)

        group_ids = torch.tensor(group_id_list, dtype=torch.long, device=device)

        # ------------------------------------------------------------------
        # 2. On-policy rollout (generation)
        # ------------------------------------------------------------------
        policy.eval()
        with _inference_autocast(device):
            generated = batch_generate(
                policy,
                tokenizer,
                all_prompts,
                max_prompt_length=max_prompt_length,
                max_new_tokens=max_completion_length,
                temperature=float(generation_cfg.get("temperature", 0.7)),
                top_p=float(generation_cfg.get("top_p", 0.9)),
                do_sample=bool(generation_cfg.get("do_sample", True)),
            )

        sequences = generated["sequences"].clone().detach().to(device)
        response_mask = generated["response_mask"].clone().detach().to(device)
        responses = generated["responses"]
        response_lengths = generated["response_lengths"]

        # Use the correct truncation flags from batch_generate, which properly
        # checks whether each response reached max_new_tokens without emitting EOS.
        # The previous hand-rolled check inspected sequences[i, -1] which points at
        # a PAD token in padded batches, causing almost all completions to be wrongly
        # flagged as truncated → effective_mask zeroed → loss = 0 on most steps.
        truncated = generated["truncated"]

        if mask_truncated:
            effective_mask = mask_truncated_sequences(response_mask, truncated)
        else:
            effective_mask = response_mask.clone()

        # ------------------------------------------------------------------
        # 3. Score completions with the frozen reward model
        # ------------------------------------------------------------------
        with _inference_autocast(device):
            raw_rewards = score_reward_pairs(
                reward_model,
                reward_tokenizer,
                all_prompts,
                responses,
                max_length=reward_max_length,
            )
        rewards = raw_rewards.detach().clone().to(dtype=torch.float32, device=device)

        # ------------------------------------------------------------------
        # 4. Old policy logprobs and reference policy logprobs
        # ------------------------------------------------------------------
        with torch.no_grad():
            old_logp, _ = response_token_logprobs(
                policy,
                sequences,
                generated["attention_mask"],
                generated["prompt_width"],
                generated["response_ids"],
            )
            old_logp = old_logp.float()

            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(
                    policy,
                    sequences,
                    generated["attention_mask"],
                    generated["prompt_width"],
                    generated["response_ids"],
                )
                ref_logp = ref_logp.float()

        # ------------------------------------------------------------------
        # 5. Group-relative advantages and informativeness statistics
        # ------------------------------------------------------------------
        advantages = group_relative_advantages(rewards, group_ids, eps=1e-6)
        group_stats = compute_group_statistics(rewards, group_ids, zero_std_tol=1e-5)

        # ------------------------------------------------------------------
        # 6. Policy optimization
        # ------------------------------------------------------------------
        policy.train()
        step_loss = 0.0
        step_diag = {}
        grad_norm_val = 0.0

        for epoch in range(policy_epochs):
            optimizer.zero_grad()
            new_logp, _ = response_token_logprobs(
                policy,
                sequences,
                generated["attention_mask"],
                generated["prompt_width"],
                generated["response_ids"],
            )
            loss, diag = grpo_policy_loss(
                new_logp=new_logp,
                old_logp=old_logp,
                seq_adv=advantages,
                token_mask=effective_mask,
                ref_logp=ref_logp,
                eps=clip_eps,
                beta=beta_kl,
                loss_type=loss_type,
                max_completion_length=max_completion_length,
            )
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm)
            optimizer.step()

            step_loss = float(loss.detach().item())
            step_diag = {k: float(v.item()) for k, v in diag.items()}
            grad_norm_val = float(gn.item() if isinstance(gn, torch.Tensor) else gn)

        clear_gpu()

        step_elapsed = time.time() - step_start
        mean_len = float(sum(response_lengths) / max(len(response_lengths), 1))
        step_entry = {
            "step": step,
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std(unbiased=False).item()),
            "within_group_reward_std": group_stats["mean_within_group_std"],
            "uninformative_group_fraction": group_stats["uninformative_group_fraction"],
            "policy_loss": step_loss,
            "policy_term": step_diag.get("policy_term", 0.0),
            "sampled_kl": step_diag.get("sampled_kl", 0.0),
            "clip_fraction": step_diag.get("clip_fraction", 0.0),
            "ratio_mean": step_diag.get("ratio_mean", 1.0),
            "sample_entropy": step_diag.get("sample_entropy", 0.0),
            "grad_norm": grad_norm_val,
            "mean_response_length": mean_len,
            "truncated_count": int(sum(truncated)),
            "step_time_seconds": round(step_elapsed, 3),
        }
        trajectory.append(step_entry)

        kl_display = f"{step_entry['sampled_kl']:.5f}" if step_entry['sampled_kl'] >= 1e-4 else f"{step_entry['sampled_kl']:.2e}"
        print(
            f"Step {step:2d}/{total_updates} | "
            f"R={step_entry['reward_mean']:+.3f} | "
            f"within_std={step_entry['within_group_reward_std']:.3f} | "
            f"uninf={step_entry['uninformative_group_fraction']:.2f} | "
            f"KL={kl_display} | "
            f"loss={step_entry['policy_loss']:.4f} | "
            f"clip={step_entry['clip_fraction']:.3f} | "
            f"grad_norm={step_entry['grad_norm']:.4f} | "
            f"entropy={step_entry['sample_entropy']:.4f} | "
            f"len={mean_len:.1f} | "
            f"t={step_elapsed:.1f}s"
        )

    total_time = time.time() - start_total_time
    peak_vram = get_peak_memory_mb(device)
    print(f"\n[GRPO Done] Total time: {total_time:.1f}s | Peak VRAM: {peak_vram:.1f} MB")

    # ------------------------------------------------------------------
    # 7. Save outputs and trajectory
    # ------------------------------------------------------------------
    policy.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    print(f"Saved policy adapter to {out_path}")

    traj_path = results_dir / f"{run_name}_trajectory.json"
    summary_data = {
        "run_name": run_name,
        "config_path": config_path,
        "loss_type": loss_type,
        "total_updates": total_updates,
        "clip_epsilon": clip_eps,
        "kl_beta": beta_kl,
        "k_generations": k_generations,
        "total_time_seconds": round(total_time, 2),
        "peak_vram_mb": round(peak_vram, 2),
        "trajectory": trajectory,
    }
    traj_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"Saved trajectory to {traj_path}")

    return {
        "output": str(out_path),
        "trajectory_file": str(traj_path),
        "total_updates": total_updates,
        "total_time_seconds": total_time,
        "peak_vram_mb": peak_vram,
    }


def main():
    ap = argparse.ArgumentParser(description="Run GRPO continuation updates.")
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output", help="Optional override for output adapter directory")
    ap.add_argument("--updates", type=int, help="Optional override for update steps")
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--clip-eps", type=float, help="Optional override for clip epsilon")
    ap.add_argument("--kl-beta", type=float, help="Optional override for KL beta")
    args = ap.parse_args()

    run_grpo(
        config_path=args.config,
        output=args.output,
        updates=args.updates,
        loss_type=args.loss_type,
        run_name=args.run_name,
        clip_epsilon=args.clip_eps,
        kl_beta=args.kl_beta,
    )


if __name__ == "__main__":
    main()
