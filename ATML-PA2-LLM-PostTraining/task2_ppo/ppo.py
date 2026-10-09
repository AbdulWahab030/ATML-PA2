"""task2_ppo/ppo.py — Core PPO objective functions for Task 2.

This module implements the token-level PPO objective used in the RLHF
continuation experiments.  Every function is validated against the
mathematical definitions given in the assignment manual:

  • GAE:   δ_t = r_t + γ V(s_{t+1}) − V(s_t)
            A^GAE_t = Σ_{k≥0} (γλ)^k δ_{t+k}

  • Shaped reward:
            r_t = r_task · 1[t=T] − β_KL · (log π_θ(a_t|s_t) − log π_ref(a_t|s_t))

  • Clipped surrogate:
            L_clip(θ) = E[min(ρ_t A_t, clip(ρ_t, 1−ε, 1+ε) A_t)]

  • Value MSE:  L_V = E[(V_φ(s_t) − R_t)²]

All arithmetic is promoted to float32 for numerical stability regardless of
the model's storage dtype.

Deliberate-defect note (Task 2)
---------------------------------
The released starter computed ``returns = advantages + values`` without
zeroing the value baseline at *padding* positions.  Because the value model
can predict non-zero scalars even for pad tokens, unmasked values leaked into
the return targets at those positions.  Although ``value_mse_loss`` applies a
mask, returning dirty targets creates inconsistencies when advantages are
computed from them.  The corrected version masks values before adding:
``returns = advantages + values * mask`` so returns at padding positions
are identically zero and consistent with the zero-masked advantages.
"""

from __future__ import annotations

import torch

from common.metrics import masked_mean


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute token-level Generalised Advantage Estimation (GAE).

    Implements the recursive form of GAE (Schulman et al., 2015):

        δ_t = r_t + γ · V(s_{t+1}) − V(s_t)
        A^GAE_t = δ_t + (γλ) · A^GAE_{t+1}

    Padding positions (``mask == 0``) are zeroed out so they do not
    propagate into valid positions when the loop unrolls backwards.
    Returns are computed as ``R_t = A^GAE_t + V(s_t)`` with the value
    baseline zeroed at padding positions for consistency.

    Args:
        rewards: ``[batch, response_steps]`` shaped reward signals
                 (already KL-shaped via :func:`shaped_rewards`).
        values:  ``[batch, response_steps]`` value-model predictions V(s_t).
        mask:    ``[batch, response_steps]`` binary validity mask
                 (1 = real response token, 0 = padding).
        gamma:   Temporal discount factor (default 1.0; typical for LLM-RL).
        lam:     GAE λ for bias–variance trade-off (default 0.95).

    Returns:
        advantages: ``[batch, response_steps]`` GAE advantages, zero at pads.
        returns:    ``[batch, response_steps]`` empirical returns (value
                    targets), zero at padding positions.
    """
    rewards = torch.nan_to_num(rewards.float(), nan=0.0, posinf=0.0, neginf=0.0)
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mask = mask.float()

    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(0)
    if values.dim() == 1:
        values = values.unsqueeze(0)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)

    min_len = min(rewards.shape[1], values.shape[1], mask.shape[1])
    rewards = rewards[:, :min_len]
    values = values[:, :min_len]
    mask = mask[:, :min_len]

    batch, steps = rewards.shape
    advantages = torch.zeros_like(rewards)
    # Running accumulator initialised to zero (corresponds to A_{T+1} = 0).
    gae = torch.zeros(batch, device=rewards.device, dtype=torch.float32)

    for t in reversed(range(steps)):
        current_valid = mask[:, t]   # [batch] — 1 iff token t is real

        if t + 1 < steps:
            next_valid = mask[:, t + 1]                  # [batch]
            # Bootstrap only from the next *valid* value; zero for terminal.
            next_value = values[:, t + 1] * next_valid   # [batch]
        else:
            next_valid = torch.zeros_like(current_valid)
            next_value = torch.zeros_like(gae)

        # TD(0) residual: δ_t = r_t + γ V(s_{t+1}) − V(s_t)
        delta = rewards[:, t] + gamma * next_value - values[:, t]

        # Recursive GAE: A_t = δ_t + (γλ) A_{t+1}
        # Multiply by next_valid so that A_{t+1} = 0 for terminal tokens.
        gae = delta + gamma * lam * next_valid * gae
        # Zero advantage at padding positions.
        gae = gae * current_valid
        advantages[:, t] = gae

    # FIX (deliberate-defect correction): mask value baseline so that return
    # targets are identically zero at padding positions, matching advantages.
    # The original starter used ``returns = advantages + values`` which left
    # raw (non-zero) critic predictions at pad positions in the targets.
    masked_values = values * mask
    returns = advantages + masked_values
    advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
    returns = torch.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    return advantages, returns


# ---------------------------------------------------------------------------
# Shaped rewards
# ---------------------------------------------------------------------------

def shaped_rewards(
    task_reward: torch.Tensor,
    policy_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    response_mask: torch.Tensor,
    beta_kl: float,
) -> torch.Tensor:
    """Apply per-token KL shaping and add terminal task reward.

    Implements the shaped reward from Ouyang et al. (2022):

        r_t = r_task · 1[t=T] − β_KL · (log π_θ(a_t|s_t) − log π_ref(a_t|s_t))

    The KL term penalises deviation from the reference policy at every
    valid token.  The task reward is added only at the last valid response
    token (t = T, the first EOS position or the response length for truncated
    sequences).

    Args:
        task_reward:   ``[batch]`` scalar reward-model scores.
        policy_logp:   ``[batch, response_steps]`` per-token log-probs under π_θ.
        ref_logp:      ``[batch, response_steps]`` per-token log-probs under π_ref.
        response_mask: ``[batch, response_steps]`` binary validity mask.
        beta_kl:       KL penalty coefficient β_KL ≥ 0.

    Returns:
        shaped: ``[batch, response_steps]`` per-token shaped rewards; zero
                at padding positions.
    """
    policy_logp = torch.nan_to_num(policy_logp.float(), nan=-100.0, posinf=0.0, neginf=-100.0)
    ref_logp = torch.nan_to_num(ref_logp.float(), nan=-100.0, posinf=0.0, neginf=-100.0)
    response_mask = response_mask.float()
    task_reward = torch.nan_to_num(task_reward.float(), nan=0.0, posinf=0.0, neginf=0.0)

    if policy_logp.dim() == 1:
        policy_logp = policy_logp.unsqueeze(0)
    if ref_logp.dim() == 1:
        ref_logp = ref_logp.unsqueeze(0)
    if response_mask.dim() == 1:
        response_mask = response_mask.unsqueeze(0)
    if task_reward.dim() == 0:
        task_reward = task_reward.unsqueeze(0)

    # Align sequence length across dimensions
    min_len = min(policy_logp.shape[1], ref_logp.shape[1], response_mask.shape[1])
    policy_logp = policy_logp[:, :min_len]
    ref_logp = ref_logp[:, :min_len]
    response_mask = response_mask[:, :min_len]

    # Per-token KL penalty: −β_KL · (log π − log π_ref) at every valid token.
    kl_diff = policy_logp - ref_logp
    kl_diff = torch.clamp(kl_diff, min=-50.0, max=50.0)
    shaped = -float(beta_kl) * kl_diff * response_mask

    # Add terminal task reward at the *last valid* response token per sequence.
    # response_mask is right-padded (1s from 0 … n-1, then 0s), so
    # sum() gives the total valid length and valid_len-1 is the last 1-index.
    for b in range(shaped.shape[0]):
        valid_len = int(response_mask[b].sum().item())
        if valid_len > 0:
            shaped[b, valid_len - 1] += task_reward[b]

    return torch.nan_to_num(shaped, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Clipped policy loss
# ---------------------------------------------------------------------------

def ppo_policy_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantage: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clipped PPO policy loss and per-update diagnostics.

    Implements the clipped surrogate objective from Schulman et al. (2017):

        L_clip(θ) = E[min(ρ_t · A_t,  clip(ρ_t, 1−ε, 1+ε) · A_t)]

    where ρ_t = π_θ(a_t|s_t) / π_old(a_t|s_t) is the importance ratio.

    ``torch.minimum`` picks the *pessimistic lower bound* in both the
    positive-advantage (cap large policy steps) and negative-advantage
    (cap large negative steps) regimes.  The scalar training loss is the
    negative masked mean (gradient ascent on L_clip ↔ gradient descent on
    −L_clip).

    Args:
        new_logp:   ``[batch, steps]`` log-probs under current policy π_θ.
        old_logp:   ``[batch, steps]`` log-probs under rollout policy π_old
                    (must be detached — treated as constants).
        advantage:  ``[batch, steps]`` normalised advantage estimates.
        mask:       ``[batch, steps]`` binary validity mask.
        eps:        Clipping threshold ε ∈ (0, 1).  Default 0.20.

    Returns:
        loss:          Scalar training loss  = −masked_mean(L_clip).
        ratio:         ``[batch, steps]`` importance ratios ρ_t (detached).
        clip_fraction: Scalar fraction of valid tokens with ρ outside
                       ``[1−ε, 1+ε]``.
    """
    new_logp = torch.nan_to_num(new_logp.float(), nan=-100.0, posinf=0.0, neginf=-100.0)
    old_logp = torch.nan_to_num(old_logp.float(), nan=-100.0, posinf=0.0, neginf=-100.0)
    advantage = torch.nan_to_num(advantage.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mask = mask.float()

    # Clamp log-ratio before exponentiation to avoid extreme exponentiation values before clipping.
    log_ratio = torch.clamp(new_logp - old_logp, min=-20.0, max=20.0)
    log_ratio = torch.nan_to_num(log_ratio, nan=0.0, posinf=20.0, neginf=-20.0)
    ratio = torch.exp(log_ratio)          # ρ_t = π_θ / π_old
    ratio = torch.nan_to_num(ratio, nan=1.0, posinf=1.0 + eps, neginf=0.0)
    ratio = torch.clamp(ratio, min=0.0, max=10.0)

    surr1 = ratio * advantage                               # unclipped
    surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * advantage  # clipped ratio

    # Pessimistic (conservative) lower bound for both +/− advantage signs.
    objective = torch.minimum(surr1, surr2)
    objective = torch.nan_to_num(objective, nan=0.0)

    # Gradient ascent on L_clip  →  minimise its negative.
    loss = -masked_mean(objective, mask)

    # Clip fraction: fraction of valid tokens outside [1−ε, 1+ε].
    clipped = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    clip_fraction = masked_mean(clipped, mask)

    return loss, ratio.detach(), clip_fraction.detach()


# ---------------------------------------------------------------------------
# Value loss
# ---------------------------------------------------------------------------

def value_mse_loss(
    predicted_values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    old_values: torch.Tensor | None = None,
    clip_eps: float | None = None,
) -> torch.Tensor:
    """Masked value loss with standard PPO value clipping.

    Minimises:
        L^{VF} = max((V_θ - R)², (V_old + clip(V_θ - V_old, -ε, ε) - R)²)

    when ``old_values`` and ``clip_eps`` are provided. Otherwise minimises
    standard masked MSE: (V_θ - R)².

    Args:
        predicted_values: ``[batch, steps]`` value-model predictions V_θ(s_t).
        returns:          ``[batch, steps]`` empirical returns R_t (detached).
        mask:             ``[batch, steps]`` binary validity mask.
        old_values:       ``[batch, steps]`` old value predictions V_old(s_t) (detached).
        clip_eps:         Clipping parameter ε (e.g. 0.20).

    Returns:
        Scalar value loss averaged over valid tokens.
    """
    pred = torch.nan_to_num(predicted_values.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    ret = torch.nan_to_num(returns.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    mask = mask.float()

    if old_values is not None and clip_eps is not None:
        old_v = torch.nan_to_num(old_values.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        eps = float(clip_eps)
        v_clipped = old_v + torch.clamp(pred - old_v, -eps, eps)
        v_loss_unclipped = (pred - ret) ** 2
        v_loss_clipped = (v_clipped - ret) ** 2
        v_loss = torch.max(v_loss_unclipped, v_loss_clipped)
    else:
        v_loss = (pred - ret) ** 2

    v_loss = torch.nan_to_num(v_loss, nan=0.0, posinf=1e4, neginf=0.0)
    return masked_mean(v_loss, mask)


value_loss = value_mse_loss


# ---------------------------------------------------------------------------
# Advantage normalisation
# ---------------------------------------------------------------------------

def normalize_advantages(
    advantages: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalise advantages across the batch: (advantages - advantages.mean()) / (advantages.std() + 1e-8).

    Normalisation statistics are computed across all valid response
    tokens in the batch.  Padding positions are zeroed out
    after normalisation.

    Args:
        advantages: ``[batch, steps]`` raw GAE advantages.
        mask:       ``[batch, steps]`` binary validity mask (optional).
        eps:        Small constant to prevent division by zero (default 1e-8).

    Returns:
        ``[batch, steps]`` normalised advantages; padding positions are 0.
    """
    advantages = torch.nan_to_num(advantages.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if mask is not None:
        mask = mask.float()
        valid = advantages[mask.bool()]
        if valid.numel() <= 1:
            return advantages * mask

        mean = valid.mean()
        std = valid.std(unbiased=False)
        normalized = (advantages - mean) / (std + eps)
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return normalized * mask
    else:
        mean = advantages.mean()
        std = advantages.std(unbiased=False)
        normalized = (advantages - mean) / (std + eps)
        return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
