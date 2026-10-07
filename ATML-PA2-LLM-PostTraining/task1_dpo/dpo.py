from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    The DPO objective is:
        loss = -log sigmoid(beta * ((log pi(y+) - log pi(y-)) - (log pi_ref(y+) - log pi_ref(y-))))
    """

    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp

    # The reference term is subtracted because the model is optimizing the logit
    # margin relative to the reference policy, not in addition to it.
    logits = beta * (policy_margin - ref_margin)

    loss = -F.logsigmoid(logits).mean()

    return loss, {
        "logit_mean": logits.detach().mean(),
        "policy_margin_mean": policy_margin.detach().mean(),
        "preference_accuracy": ((policy_margin - ref_margin) > 0).float().mean().detach(),
    }
