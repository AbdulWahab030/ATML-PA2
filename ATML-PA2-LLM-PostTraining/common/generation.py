from __future__ import annotations

import torch
import torch.nn.functional as F


try:
    from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
except ImportError:
    class LogitsProcessor:
        pass
    class LogitsProcessorList(list):
        def __call__(self, input_ids, scores):
            for processor in self:
                scores = processor(input_ids, scores)
            return scores


class SafeLogitsProcessor(LogitsProcessor):
    """Logits processor that sanitizes non-finite values and clamps logits before sampling."""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nan_to_num(scores.float(), nan=-1e4, posinf=1e4, neginf=-1e4)
        return torch.clamp(scores, min=-50.0, max=50.0)


# Safe multinomial wrapper to guarantee no NaN, Inf, or negative probabilities reach torch.multinomial
_orig_multinomial = torch.multinomial


def safe_multinomial(
    input: torch.Tensor,
    num_samples: int,
    replacement: bool = False,
    *,
    generator: torch.Generator | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Clamp probabilities and remove NaNs/Infs before calling PyTorch sampling."""
    probs = input.float()
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = torch.clamp(probs, min=0.0, max=1.0)
    prob_sum = probs.sum(dim=-1, keepdim=True)
    zero_mask = prob_sum <= 0.0
    if zero_mask.any():
        fallback = torch.ones_like(probs) / max(1, probs.shape[-1])
        probs = torch.where(zero_mask, fallback, probs / prob_sum.clamp_min(1e-8))
    else:
        probs = probs / prob_sum.clamp_min(1e-8)
    return _orig_multinomial(probs, num_samples, replacement=replacement, generator=generator, out=out)


torch.multinomial = safe_multinomial


def _response_mask(response_ids: torch.Tensor, eos_id: int | None) -> torch.Tensor:
    mask = torch.ones_like(response_ids, dtype=torch.float32)
    if eos_id is None:
        return mask
    for i in range(response_ids.shape[0]):
        eos_pos = (response_ids[i] == eos_id).nonzero(as_tuple=False)
        if len(eos_pos):
            first = int(eos_pos[0].item())
            if first + 1 < response_ids.shape[1]:
                mask[i, first + 1:] = 0.0
    return mask


def batch_generate(
    model,
    tokenizer,
    prompts: list[list[dict]],
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
):
    rendered = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    enc = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    was_training = model.training
    model.eval()
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "logits_processor": LogitsProcessorList([SafeLogitsProcessor()]),
    }
    if do_sample:
        kwargs.update({"temperature": temperature, "top_p": top_p})

    with torch.no_grad():
        seq = model.generate(**enc, **kwargs)
    seq = seq.clone().detach()
    if was_training:
        model.train()

    prompt_width = enc["input_ids"].shape[1]
    response_ids = seq[:, prompt_width:].clone().detach()
    rmask = _response_mask(response_ids, tokenizer.eos_token_id).to(seq.device)
    terminated, truncated, lengths, texts = [], [], [], []
    for row, mask in zip(response_ids, rmask):
        n = int(mask.sum().item())
        has_eos = bool((row[:n] == tokenizer.eos_token_id).any().item()) if tokenizer.eos_token_id is not None else False
        terminated.append(has_eos)
        truncated.append(bool(n >= max_new_tokens and not has_eos))
        lengths.append(n)
        texts.append(tokenizer.decode(row[:n], skip_special_tokens=True))

    attn = torch.ones_like(seq, dtype=torch.long)
    attn[:, :prompt_width] = enc["attention_mask"]
    return {
        "sequences": seq,
        "attention_mask": attn,
        "prompt_width": prompt_width,
        "response_ids": response_ids,
        "response_mask": rmask,
        "responses": texts,
        "terminated_with_eos": terminated,
        "truncated": truncated,
        "response_lengths": lengths,
    }


def response_token_logprobs(model, sequences, attention_mask, prompt_width, response_ids):
    outputs = model(
        input_ids=sequences,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    logits = outputs.logits[:, prompt_width - 1 : -1, :]
    logits = logits[:, : response_ids.shape[1], :]
    # Safeguards: sanitize non-finite values and clamp logits before log_softmax
    logits = torch.nan_to_num(logits.float(), nan=-1e4, posinf=1e4, neginf=-1e4)
    logits = torch.clamp(logits, min=-100.0, max=100.0)
    logp = logits.log_softmax(dim=-1)
    logp = torch.nan_to_num(logp, nan=-100.0, posinf=0.0, neginf=-100.0)
    chosen = torch.gather(logp, -1, response_ids.unsqueeze(-1)).squeeze(-1)
    chosen = torch.nan_to_num(chosen, nan=-100.0, posinf=0.0, neginf=-100.0)
    return chosen, logits


def response_sequence_logprobs(model, batch: dict):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    logits = outputs.logits[:, :-1, :]
    labels = batch["input_ids"][:, 1:]
    mask = batch["response_mask"][:, 1:]
    # Safeguards: sanitize non-finite values and clamp logits before log_softmax
    logits = torch.nan_to_num(logits.float(), nan=-1e4, posinf=1e4, neginf=-1e4)
    logits = torch.clamp(logits, min=-100.0, max=100.0)
    logp = logits.log_softmax(dim=-1)
    logp = torch.nan_to_num(logp, nan=-100.0, posinf=0.0, neginf=-100.0)
    tok = torch.gather(logp, -1, labels.unsqueeze(-1)).squeeze(-1)
    tok = torch.nan_to_num(tok, nan=-100.0, posinf=0.0, neginf=-100.0)
    return (tok * mask).sum(-1), tok, mask


@torch.no_grad()
def score_reward_pairs(rm_model, rm_tokenizer, prompts, responses, max_length=1024):
    texts = []
    for prompt, response in zip(prompts, responses):
        messages = list(prompt) + [{"role": "assistant", "content": response}]
        texts.append(
            rm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )

    device = next(rm_model.parameters()).device
    enc = rm_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    return rm_model(**enc).logits[:, 0].float()
