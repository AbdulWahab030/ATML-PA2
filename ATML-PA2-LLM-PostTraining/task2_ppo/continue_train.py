"""task2_ppo/continue_train.py — Standard PPO continuation from the course midpoint.

Entry-point command (run from the repository root)::

    python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard

This script:
1. Loads the supplied midpoint policy and value checkpoints.
2. Runs ``total_updates`` online rollout-and-update cycles.
3. Logs the required trajectory diagnostics to a JSON file under
   ``results/task2_ppo/``.
4. Saves the final policy adapter to ``outputs/task2_ppo/<run_name>/``.

Cross-platform device handling
---------------------------------
* Device is selected dynamically: CUDA → MPS → CPU.
* ``torch.autocast`` is used for the reward-model inference pass only,
  with a safe fallback when the device does not support it.
* Memory stats (reset/peak query) branch on device type.
* ``clear_gpu`` (from common.models) handles both ``torch.cuda.empty_cache``
  and ``torch.mps.empty_cache`` with graceful fallbacks.
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
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


# ---------------------------------------------------------------------------
# Cross-platform device utilities
# ---------------------------------------------------------------------------

def _detect_device() -> torch.device:
    """Return the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _reset_peak_memory(device: torch.device) -> None:
    """Reset peak-memory tracking if the device supports it."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # MPS does not expose a peak-reset API; nothing to do.


def get_peak_memory_mb(device: torch.device) -> float:
    """Return peak allocated device memory in megabytes.

    Returns 0.0 on devices that do not expose memory-tracking APIs.

    Args:
        device: The active compute device.

    Returns:
        Peak allocated memory in MB, or 0.0 if unavailable.
    """
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
    """Context manager for mixed-precision inference.

    Wraps ``torch.autocast`` on CUDA/MPS; falls back to a no-op on CPU
    or when the device does not fully support autocast.

    Args:
        device: Active compute device.

    Yields:
        An autocast context (or a no-op context on unsupported devices).
    """
    if device.type in {"cuda", "mps"}:
        try:
            with torch.autocast(device_type=device.type, enabled=True):
                yield
            return
        except RuntimeError:
            pass  # fall through to no-op
    # CPU or unsupported: run without autocast.
    with contextlib.nullcontext():
        yield


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def prepare_ppo_continuation(config_path: str) -> dict:
    """Load models, tokenizer, prompt pool, and optimisers for PPO continuation.

    Reads the merged YAML config (``ppo.yaml`` inheriting ``base.yaml``),
    loads the supplied midpoint policy and value checkpoints, and constructs
    separate AdamW optimisers for policy LoRA and critic LoRA+head parameters.

    Args:
        config_path: Path to the task-specific YAML config (e.g.
                     ``configs/ppo.yaml``).

    Returns:
        A dict with keys: ``cfg``, ``tokenizer``, ``policy``,
        ``value_model``, ``reward_model``, ``reward_tokenizer``,
        ``prompt_rows``, ``policy_optimizer``, ``value_optimizer``.
    """
    cfg = load_yaml(config_path)
    set_seed(int(cfg.get("seed", 6304)))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "lora_head"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_ppo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    clip_epsilon: float | None = None,
    kl_beta: float | None = None,
    run_name: str = "standard",
) -> dict:
    """Run online PPO continuation updates from the supplied midpoint checkpoint.

    Implements the standard PPO-RLHF loop:

    For each update step:
      1. **Rollout** — generate on-policy responses with the current policy.
      2. **Reward** — score responses with the frozen reward model; apply
         missing-EOS penalty.
      3. **Old logprobs / values** — compute π_old and π_ref log-probs,
         and old value estimates, all under ``torch.no_grad()``.
      4. **GAE** — compute shaped rewards, advantages, and empirical returns.
      5. **PPO epochs** — for each inner epoch, recompute π_θ log-probs,
         apply the clipped policy loss, update the policy; then update the
         value model with MSE loss.
      6. **Logging** — record all required trajectory diagnostics.

    Args:
        config_path:  Path to ``configs/ppo.yaml``.
        output:       Override output directory for the saved policy adapter.
        updates:      Override number of continuation update steps.
        clip_epsilon: Override clipping parameter ε.
        kl_beta:      Override KL shaping coefficient β_KL.
        run_name:     Identifier used in log file names and the output path.

    Returns:
        A dict with keys ``output``, ``trajectory_file``,
        ``total_updates``, ``total_time_seconds``, ``peak_vram_mb``.
    """
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]

    # ---------- device selection (cross-platform) ----------
    device = _detect_device()
    print(f"[device] Using: {device}")

    # Move trainable models to device.
    policy = bundle["policy"].to(device)
    value_model = bundle["value_model"].to(device)
    tokenizer = bundle["tokenizer"]
    reward_model = bundle["reward_model"]       # reward model manages its own device via device_map
    reward_tokenizer = bundle["reward_tokenizer"]
    prompt_rows = bundle["prompt_rows"]
    policy_optimizer = bundle["policy_optimizer"]
    value_optimizer = bundle["value_optimizer"]

    # ---------- hyperparameters from config / overrides ----------
    total_updates = int(updates if updates is not None else cfg.get("updates", 20))
    clip_eps = float(clip_epsilon if clip_epsilon is not None else cfg.get("clip_epsilon", 0.20))
    beta_kl = float(kl_beta if kl_beta is not None else cfg.get("kl_beta", 0.10))
    out_path = repo_path(output or cfg.get("output", f"outputs/task2_ppo/{run_name}"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    ppo_epochs = int(cfg.get("ppo_epochs", 2))
    prompts_per_update = int(cfg.get("prompts_per_update", 1))
    gamma = float(cfg.get("gamma", 1.0))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))
    missing_eos_penalty = float(cfg.get("missing_eos_penalty", 1.0))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    max_prompt_length = int(cfg.get("max_prompt_length", 256))
    max_response_length = int(cfg.get("max_response_length", 512))
    reward_max_length = int(cfg.get("reward_max_length", 1280))
    generation_cfg = cfg.get("generation", {})

    print(f"=== Starting PPO Continuation: {run_name} ===")
    print(f"Updates: {total_updates}  clip_eps: {clip_eps}  beta_kl: {beta_kl}  ppo_epochs: {ppo_epochs}")
    print(f"Output adapter: {out_path}")

    trajectory: list[dict] = []
    start_total_time = time.time()
    num_prompts = len(prompt_rows)

    _reset_peak_memory(device)

    for step in range(1, total_updates + 1):
        step_start = time.time()

        # ------------------------------------------------------------------
        # 1.  Select prompts for this update (cyclic over the prompt pool).
        # ------------------------------------------------------------------
        p_start = ((step - 1) * prompts_per_update) % num_prompts
        p_end = p_start + prompts_per_update
        if p_end <= num_prompts:
            batch_rows = prompt_rows[p_start:p_end]
        else:
            batch_rows = prompt_rows[p_start:] + prompt_rows[: p_end % num_prompts]
        prompts = [prompt_messages(r) for r in batch_rows]

        # ------------------------------------------------------------------
        # 2.  On-policy rollout (generation).
        # ------------------------------------------------------------------
        policy.eval()
        value_model.eval()
        generated = batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_response_length,
            temperature=float(generation_cfg.get("temperature", 0.7)),
            top_p=float(generation_cfg.get("top_p", 0.9)),
            do_sample=bool(generation_cfg.get("do_sample", True)),
        )

        sequences = generated["sequences"].clone().detach()
        attention_mask = generated["attention_mask"].clone().detach()
        prompt_width = generated["prompt_width"]
        response_ids = generated["response_ids"].clone().detach()
        response_mask = generated["response_mask"].clone().detach().to(device)
        responses = generated["responses"]
        terminated = generated["terminated_with_eos"]

        # ------------------------------------------------------------------
        # 3.  Reward-model scoring with missing-EOS penalty.
        #     Use autocast for reward model inference (frozen, fp16/bf16-safe).
        # ------------------------------------------------------------------
        with _inference_autocast(device):
            raw_rewards = score_reward_pairs(
                reward_model,
                reward_tokenizer,
                prompts,
                responses,
                max_length=reward_max_length,
            ).to(device)

        task_rewards = raw_rewards.clone()
        for b, has_eos in enumerate(terminated):
            if not has_eos:
                task_rewards[b] -= missing_eos_penalty

        # ------------------------------------------------------------------
        # 4.  Rollout log-probs (old policy), reference log-probs, old values.
        #     All computed with no_grad — these are the fixed rollout stats.
        # ------------------------------------------------------------------
        with torch.no_grad():
            old_logp, _ = response_token_logprobs(
                policy, sequences, attention_mask, prompt_width, response_ids
            )
            old_logp = old_logp.clone().detach()

            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(
                    policy, sequences, attention_mask, prompt_width, response_ids
                )
            ref_logp = ref_logp.clone().detach()

            # Value predictions at response positions.
            val_all = token_values(value_model, sequences, attention_mask)
            # Slice to response window: positions [prompt_width-1 … -1].
            val_response = val_all[:, prompt_width - 1: -1]
            old_values = val_response[:, : response_ids.shape[1]].clone().detach()

        # ------------------------------------------------------------------
        # 5.  Compute shaped rewards, GAE advantages, and empirical returns.
        # ------------------------------------------------------------------
        with torch.no_grad():
            shaped_rew = shaped_rewards(
                task_rewards, old_logp, ref_logp, response_mask, beta_kl
            )
            advantages, returns = compute_gae(
                shaped_rew, old_values, response_mask, gamma=gamma, lam=gae_lambda
            )
            norm_adv = normalize_advantages(advantages, response_mask).clone().detach()
            returns = returns.clone().detach()

        # ------------------------------------------------------------------
        # 6.  PPO update epochs (policy + value).
        # ------------------------------------------------------------------
        policy_loss_acc = 0.0
        value_loss_acc = 0.0
        clip_frac_acc = 0.0
        entropy_acc = 0.0
        pi_grad_norm_acc = 0.0
        v_grad_norm_acc = 0.0

        for _epoch in range(ppo_epochs):
            # ---- Policy update ----
            policy.train()
            policy_optimizer.zero_grad(set_to_none=True)

            new_logp, _ = response_token_logprobs(
                policy, sequences, attention_mask, prompt_width, response_ids
            )
            loss_pi, _, clip_frac = ppo_policy_loss(
                new_logp,
                old_logp.detach(),
                norm_adv.detach(),
                response_mask,
                eps=clip_eps,
            )
            loss_pi.backward()
            pi_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters(policy), max_norm=max_grad_norm
            )
            policy_optimizer.step()

            # ---- Value update ----
            value_model.train()
            value_optimizer.zero_grad(set_to_none=True)

            new_val_all = token_values(value_model, sequences, attention_mask)
            pred_values = new_val_all[:, prompt_width - 1: -1][:, : response_ids.shape[1]]
            loss_v = value_mse_loss(pred_values, returns.detach(), response_mask)
            loss_v.backward()
            v_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters(value_model), max_norm=max_grad_norm
            )
            value_optimizer.step()

            # Accumulate diagnostics across PPO epochs.
            policy_loss_acc += float(loss_pi.detach().item())
            value_loss_acc += float(loss_v.detach().item())
            clip_frac_acc += float(clip_frac.detach().item())
            entropy_acc += float(sample_entropy(new_logp.detach(), response_mask).item())
            pi_grad_norm_acc += float(pi_norm.item() if isinstance(pi_norm, torch.Tensor) else pi_norm)
            v_grad_norm_acc += float(v_norm.item() if isinstance(v_norm, torch.Tensor) else v_norm)

        # Average across PPO epochs.
        policy_loss_mean = policy_loss_acc / ppo_epochs
        value_loss_mean = value_loss_acc / ppo_epochs
        clip_frac_mean = clip_frac_acc / ppo_epochs
        entropy_mean = entropy_acc / ppo_epochs
        pi_grad_norm_mean = pi_grad_norm_acc / ppo_epochs
        v_grad_norm_mean = v_grad_norm_acc / ppo_epochs

        # ------------------------------------------------------------------
        # 7.  Logging diagnostics for this step.
        # ------------------------------------------------------------------
        with torch.no_grad():
            cur_kl = float(sampled_kl(new_logp.detach(), ref_logp.detach(), response_mask).item())
            mean_task_reward = float(task_rewards.mean().item())
            mean_raw_reward = float(raw_rewards.mean().item())
            mean_resp_len = float(mean_response_length(response_mask))

        step_elapsed = time.time() - step_start
        peak_vram = get_peak_memory_mb(device)

        record: dict = {
            "update": step,
            "raw_reward_mean": mean_raw_reward,
            "task_reward_mean": mean_task_reward,
            "kl_divergence": cur_kl,
            "policy_loss": policy_loss_mean,
            "value_loss": value_loss_mean,
            "clip_fraction": clip_frac_mean,
            "entropy": entropy_mean,
            "policy_grad_norm": pi_grad_norm_mean,
            "value_grad_norm": v_grad_norm_mean,
            "response_length_tokens": mean_resp_len,
            "step_wall_clock_seconds": step_elapsed,
            "peak_vram_mb": peak_vram,
        }
        trajectory.append(record)

        print(
            f"update={step:02d}/{total_updates} "
            f"reward={mean_task_reward:.3f} "
            f"kl={cur_kl:.4f} "
            f"pi_loss={policy_loss_mean:.4f} "
            f"v_loss={value_loss_mean:.4f} "
            f"clip={clip_frac_mean:.3f} "
            f"entropy={entropy_mean:.3f} "
            f"len={mean_resp_len:.1f} "
            f"time={step_elapsed:.1f}s"
        )

    # ---------- Save ----------
    total_time = time.time() - start_total_time
    final_vram = get_peak_memory_mb(device)
    print(f"\nContinuation complete in {total_time:.1f}s.  Peak device memory: {final_vram:.1f} MB.")

    policy.save_pretrained(out_path)
    print(f"Saved policy adapter → {out_path}")

    traj_file = results_dir / f"{run_name}_trajectory.json"
    traj_file.write_text(json.dumps(trajectory, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved trajectory log → {traj_file}")

    clear_gpu(policy, value_model, reward_model)

    return {
        "output": str(out_path),
        "trajectory_file": str(traj_file),
        "total_updates": total_updates,
        "total_time_seconds": total_time,
        "peak_vram_mb": final_vram,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry-point for the standard PPO continuation."""
    ap = argparse.ArgumentParser(
        description="Run PPO continuation from the course midpoint checkpoint."
    )
    ap.add_argument("--config", default="configs/ppo.yaml",
                    help="Path to the task PPO config (default: configs/ppo.yaml).")
    ap.add_argument("--output",
                    help="Override output adapter directory.")
    ap.add_argument("--updates", type=int,
                    help="Override number of continuation update steps.")
    ap.add_argument("--clip-epsilon", type=float,
                    help="Override clipping parameter ε.")
    ap.add_argument("--kl-beta", type=float,
                    help="Override KL shaping coefficient β_KL.")
    ap.add_argument("--run-name", default="standard",
                    help="Identifier used in log file names (default: standard).")
    args = ap.parse_args()

    run_ppo(
        config_path=args.config,
        output=args.output,
        updates=args.updates,
        clip_epsilon=args.clip_epsilon,
        kl_beta=args.kl_beta,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
