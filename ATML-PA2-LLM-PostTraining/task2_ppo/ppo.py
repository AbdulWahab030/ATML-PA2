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
    rewards = rewards.float()
    values = values.float()
    mask = mask.float()

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
    policy_logp = policy_logp.float()
    ref_logp = ref_logp.float()
    response_mask = response_mask.float()
    task_reward = task_reward.float()

    # Per-token KL penalty: −β_KL · (log π − log π_ref) at every valid token.
    shaped = -float(beta_kl) * (policy_logp - ref_logp) * response_mask

    # Add terminal task reward at the *last valid* response token per sequence.
    # response_mask is right-padded (1s from 0 … n-1, then 0s), so
    # sum() gives the total valid length and valid_len-1 is the last 1-index.
    for b in range(shaped.shape[0]):
        valid_len = int(response_mask[b].sum().item())
        if valid_len > 0:
            shaped[b, valid_len - 1] += task_reward[b]

    return shaped


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
    new_logp = new_logp.float()
    old_logp = old_logp.float()
    advantage = advantage.float()
    mask = mask.float()

    # Clamp log-ratio before exponentiation for numerical stability.
    log_ratio = torch.clamp(new_logp - old_logp, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)          # ρ_t = π_θ / π_old

    surr1 = ratio * advantage                               # unclipped
    surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * advantage  # clipped ratio

    # Pessimistic (conservative) lower bound for both +/− advantage signs.
    objective = torch.minimum(surr1, surr2)

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
) -> torch.Tensor:
    """Masked mean-squared-error loss for the value function.

    Minimises E[(V_φ(s_t) − R_t)²] over valid response tokens.
    ``returns`` must be passed **detached**; they are treated as fixed
    regression targets and should not receive gradients.

    Args:
        predicted_values: ``[batch, steps]`` value-model predictions V_φ(s_t).
        returns:          ``[batch, steps]`` empirical returns R_t (detached).
        mask:             ``[batch, steps]`` binary validity mask.

    Returns:
        Scalar MSE loss averaged over valid tokens.
    """
    pred = predicted_values.float()
    ret = returns.float()
    mask = mask.float()
    diff = pred - ret
    return masked_mean(diff ** 2, mask)


# ---------------------------------------------------------------------------
# Advantage normalisation
# ---------------------------------------------------------------------------

def normalize_advantages(
    advantages: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalise advantages to zero mean and unit variance over valid tokens.

    Normalisation statistics are computed across **all** valid response
    tokens in the batch (i.e. the token-level, not sequence-level,
    distribution is standardised).  Padding positions are zeroed out
    after normalisation.

    Args:
        advantages: ``[batch, steps]`` raw GAE advantages.
        mask:       ``[batch, steps]`` binary validity mask.
        eps:        Small constant to prevent division by zero (default 1e-6).

    Returns:
        ``[batch, steps]`` normalised advantages; padding positions are 0.
    """
    advantages = advantages.float()
    mask = mask.float()

    # Select all valid token advantages as a flat 1-D tensor.
    valid = advantages[mask.bool()]
    if valid.numel() <= 1:
        return advantages * mask

    mean = valid.mean()
    std = valid.std(unbiased=False).clamp_min(eps)
    return ((advantages - mean) / std) * mask
