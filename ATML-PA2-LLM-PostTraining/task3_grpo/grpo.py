"""task3_grpo/grpo.py — Core GRPO objective functions for Task 3.

This module implements the critic-free Group Relative Policy Optimization (GRPO)
objective (Shao et al., 2024; DeepSeekMath):

    A_k = (r_k − μ_r) / (σ_r + ε)

where for a prompt with K sampled completions:
    μ_r = (1 / K) Σ_{j=1}^K r_j
    σ_r = sqrt((1 / K) Σ_{j=1}^K (r_j − μ_r)²)

Clipped surrogate loss:
    L_GRPO(θ) = −(1 / K) Σ_{k=1}^K [ (1 / |y_k|) Σ_{t=1}^{|y_k|} min(ρ_{k,t} A_k, clip(ρ_{k,t}, 1−ε, 1+ε) A_k) ] + β D_KL(π_θ || π_ref)

Dr. GRPO normalization (Liu et al., 2025):
    Replaces (1 / |y_k|) with (1 / L_max) to remove optimization gradient bias
    against longer correct completions.

Deliberate-defect note (Task 3):
---------------------------------
The starter implementation computed a batch-wide mean and std across all completions
irrespective of prompt grouping:
    mean = rewards.mean()
    std = rewards.std(unbiased=False).clamp_min(eps)
    return (rewards - mean) / std
This destroyed the within-prompt comparative structure of GRPO. The corrected
implementation groups rewards strictly by `group_ids`, computing per-group μ_r
and σ_r, and returning (r_k − μ_r) / (σ_r + ε) for each completion.
"""

from __future__ import annotations

import torch

from common.metrics import masked_mean, sample_entropy


def group_relative_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return one scalar advantage per sampled completion normalized within its prompt group.

    Args:
        rewards:   ``[batch]`` scalar rewards for all sampled completions.
        group_ids: ``[batch]`` identifier grouping completions by prompt.
        eps:       Numerical stability constant added to group std (default 1e-6).

    Returns:
        advantages: ``[batch]`` group-relative normalized scalar advantages.
    """
    rewards = rewards.float()
    group_ids = group_ids.to(rewards.device)
    advantages = torch.zeros_like(rewards)

    unique_groups = torch.unique(group_ids)
    for gid in unique_groups:
        mask = (group_ids == gid)
        g_rewards = rewards[mask]
        g_mean = g_rewards.mean()
        # Biased (population) std matching DeepSeekMath formula (1/K):
        g_std = g_rewards.std(unbiased=False)
        advantages[mask] = (g_rewards - g_mean) / (g_std + eps)

    return advantages


def compute_group_statistics(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    zero_std_tol: float = 1e-5,
) -> dict[str, float]:
    """Compute informativeness diagnostics across prompt groups.

    A group is uninformative when the sampled completions receive identical
    (or near-identical) rewards, giving σ_r ≈ 0.

    Args:
        rewards:      ``[batch]`` scalar rewards.
        group_ids:    ``[batch]`` prompt group IDs.
        zero_std_tol: Threshold below which group std is deemed uninformative.

    Returns:
        Dictionary with:
          - ``mean_within_group_std``: average σ_r across groups
          - ``uninformative_group_fraction``: fraction of groups with σ_r <= zero_std_tol
          - ``informative_group_fraction``: 1.0 - uninformative_group_fraction
          - ``num_groups``: total unique prompt groups evaluated
    """
    rewards = rewards.float()
    group_ids = group_ids.to(rewards.device)
    unique_groups = torch.unique(group_ids)

    stds = []
    uninformative_count = 0

    for gid in unique_groups:
        mask = (group_ids == gid)
        g_rewards = rewards[mask]
        g_std = float(g_rewards.std(unbiased=False).item())
        stds.append(g_std)
        if g_std <= zero_std_tol:
            uninformative_count += 1

    num_groups = max(len(unique_groups), 1)
    mean_std = float(sum(stds) / num_groups)
    uninformative_frac = float(uninformative_count / num_groups)

    return {
        "mean_within_group_std": mean_std,
        "uninformative_group_fraction": uninformative_frac,
        "informative_group_fraction": 1.0 - uninformative_frac,
        "num_groups": len(unique_groups),
    }


def grpo_policy_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    seq_adv: torch.Tensor,
    token_mask: torch.Tensor,
    ref_logp: torch.Tensor,
    eps: float,
    beta: float,
    loss_type: str = "grpo",
    max_completion_length: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """PPO-style clipped GRPO loss for already-sampled completions.

    Args:
        new_logp:              ``[batch, seq_len]`` log-probs under current policy π_θ.
        old_logp:              ``[batch, seq_len]`` log-probs under sampling policy π_old.
        seq_adv:               ``[batch]`` scalar sequence-level advantages A_k.
        token_mask:            ``[batch, seq_len]`` binary response token validity mask.
        ref_logp:              ``[batch, seq_len]`` log-probs under reference policy π_ref.
        eps:                   Clipping parameter ε (e.g. 0.20).
        beta:                  KL penalty coefficient β (e.g. 0.10).
        loss_type:             ``'grpo'`` (1/|y_k| normalization) or
                               ``'dr_grpo'`` (1/L_max constant normalization).
        max_completion_length: Maximum completion length L_max (required for Dr. GRPO).

    Returns:
        loss:                  Scalar training objective.
        diagnostics:           Dict containing policy term, KL, clip fraction, entropy, etc.
    """
    new_logp = new_logp.float()
    old_logp = old_logp.float()
    ref_logp = ref_logp.float()
    token_mask = token_mask.float()
    seq_adv = seq_adv.float()

    # Log-ratio clamped for numerical stability before exp:
    log_ratio = torch.clamp(new_logp - old_logp, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)

    adv = seq_adv[:, None]
    s1 = ratio * adv
    s2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    objective = torch.minimum(s1, s2)

    token_sum = (objective * token_mask).sum(-1)

    if loss_type == "grpo":
        denom = token_mask.sum(-1).clamp_min(1.0)
        per_sequence = token_sum / denom
        policy_term = -per_sequence.mean()
    elif loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("dr_grpo requires max_completion_length")
        per_sequence = token_sum / float(max_completion_length)
        policy_term = -per_sequence.mean()
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")

    # Reverse KL estimator: exp(log π_ref - log π) - (log π_ref - log π) - 1
    log_ratio_ref_over_policy = ref_logp - new_logp
    per_token_kl = torch.exp(torch.clamp(log_ratio_ref_over_policy, min=-20.0, max=20.0)) - log_ratio_ref_over_policy - 1.0
    kl = masked_mean(per_token_kl, token_mask)

    loss = policy_term + float(beta) * kl

    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    return loss, {
        "policy_term": policy_term.detach(),
        "sampled_kl": kl.detach(),
        "clip_fraction": masked_mean(affected, token_mask).detach(),
        "ratio_mean": masked_mean(ratio.detach(), token_mask),
        "sample_entropy": sample_entropy(new_logp.detach(), token_mask),
    }


def mask_truncated_sequences(token_mask: torch.Tensor, truncated: list[bool] | torch.Tensor) -> torch.Tensor:
    """Mask out completions that hit the maximum generation length without terminating.

    Per the assignment specification, completions reaching max-length are excluded
    from the training objective to prevent optimizing truncated, incomplete outputs.

    Args:
        token_mask: ``[batch, seq_len]`` response validity mask.
        truncated:  ``[batch]`` boolean flags indicating truncation at max tokens.

    Returns:
        Masked token validity tensor with truncated rows zeroed.
    """
    truncated = torch.as_tensor(truncated, device=token_mask.device, dtype=torch.bool)
    keep = (~truncated).to(token_mask.dtype)[:, None]
    return token_mask * keep
