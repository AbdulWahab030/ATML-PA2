from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import set_seed
from common.metrics import word_count
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.dpo import dpo_loss


def _sequence_logprob(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    response_mask = batch["response_mask"].to(device)
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    token_log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    labels = input_ids[:, 1:]
    selected_log_probs = torch.gather(
        token_log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)
    return (selected_log_probs * response_mask[:, 1:]).sum(dim=-1)


def _preference_batch(tokenizer, rows, max_length):
    chosen, rejected = [], []
    for row in rows:
        prompt = prompt_messages_from_preference(row)
        chosen_text, rejected_text = preference_responses(row)
        chosen.append(encode_prompt_response(tokenizer, prompt, chosen_text, max_length))
        rejected.append(encode_prompt_response(tokenizer, prompt, rejected_text, max_length))
    return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)


def _to_float(value):
    return float(value.detach().cpu().item())


def evaluate(config_path: str, adapter: str, name: str = "standard", dataset_path: str | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    eval_file = dataset_path or cfg["paths"]["dpo_standard_eval"]
    rows = read_jsonl(eval_file)
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    device = next(policy.parameters()).device
    max_length = int(cfg["max_sequence_length"])
    batch_size = int(cfg.get("eval_batch_size", 2))

    # Keep prompts intact and make any excluded held-out rows visible in the report.
    valid_rows, excluded = [], 0
    for row in rows:
        prompt = prompt_messages_from_preference(row)
        chosen, rejected = preference_responses(row)
        try:
            encode_prompt_response(tokenizer, prompt, chosen, max_length)
            encode_prompt_response(tokenizer, prompt, rejected, max_length)
        except ValueError:
            excluded += 1
            continue
        valid_rows.append(row)
    if not valid_rows:
        raise ValueError(
            f"No held-out preference rows fit max_sequence_length={max_length}; "
            "increase the configured sequence length or inspect the evaluation data."
        )

    preference_losses = []
    preference_accuracies = []
    for start in range(0, len(valid_rows), batch_size):
        batch_rows = valid_rows[start : start + batch_size]
        chosen_batch, rejected_batch = _preference_batch(tokenizer, batch_rows, max_length)
        with torch.inference_mode():
            policy_chosen = _sequence_logprob(policy, chosen_batch, device)
            policy_rejected = _sequence_logprob(policy, rejected_batch, device)
            with reference_mode(policy):
                ref_chosen = _sequence_logprob(policy, chosen_batch, device)
                ref_rejected = _sequence_logprob(policy, rejected_batch, device)
            loss, diagnostics = dpo_loss(
                policy_chosen,
                policy_rejected,
                ref_chosen,
                ref_rejected,
                float(cfg["beta"]),
            )
        preference_losses.append(_to_float(loss))
        preference_accuracies.append(_to_float(diagnostics["preference_accuracy"]))

    generation_prompts = [prompt_messages_from_preference(row) for row in valid_rows]
    generation_cfg = cfg.get("generation", {})
    generated_records = []
    sampled_kl_token_sum = 0.0
    sampled_kl_token_count = 0
    reward_scores = []
    response_lengths = []
    response_words = []
    truncated_count = 0
    max_prompt_length = int(cfg.get("max_prompt_length", max_length))
    max_new_tokens = int(cfg["max_generation_tokens"])

    for start in range(0, len(valid_rows), batch_size):
        batch_rows = valid_rows[start : start + batch_size]
        prompts = generation_prompts[start : start + batch_size]
        generated = batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            temperature=float(generation_cfg.get("temperature", 0.7)),
            top_p=float(generation_cfg.get("top_p", 0.9)),
            do_sample=bool(generation_cfg.get("do_sample", True)),
        )
        with torch.inference_mode():
            policy_logp, _ = response_token_logprobs(
                policy,
                generated["sequences"],
                generated["attention_mask"],
                generated["prompt_width"],
                generated["response_ids"],
            )
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(
                    policy,
                    generated["sequences"],
                    generated["attention_mask"],
                    generated["prompt_width"],
                    generated["response_ids"],
                )
        mask = generated["response_mask"].to(policy_logp.device)
        sampled_kl_token_sum += float(((policy_logp - ref_logp) * mask).sum().cpu())
        sampled_kl_token_count += int(mask.sum().cpu())

        scores = score_reward_pairs(
            reward_model,
            reward_tokenizer,
            prompts,
            generated["responses"],
        )
        batch_scores = [float(score) for score in scores.detach().cpu().tolist()]
        reward_scores.extend(batch_scores)
        response_lengths.extend(int(n) for n in generated["response_lengths"])
        response_words.extend(word_count(text) for text in generated["responses"])
        truncated_count += sum(generated["truncated"])

        for offset, (row, prompt, response) in enumerate(
            zip(batch_rows, prompts, generated["responses"])
        ):
            chosen, rejected = preference_responses(row)
            generated_records.append(
                {
                    "row_index": start + offset,
                    "id": row.get("id", row.get("prompt_id")),
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "generated": response,
                    "generated_reward_score": batch_scores[offset],
                    "generated_tokens": int(generated["response_lengths"][offset]),
                    "generated_words": word_count(response),
                    "truncated": bool(generated["truncated"][offset]),
                }
            )

    count = len(valid_rows)
    summary = {
        "name": name,
        "adapter": str(adapter),
        "evaluation_dataset": str(eval_file),
        "num_rows_total": len(rows),
        "num_rows_evaluated": count,
        "num_rows_excluded_for_length": excluded,
        "beta": float(cfg["beta"]),
        "heldout_dpo_loss": sum(preference_losses) / len(preference_losses),
        "heldout_preference_accuracy": sum(preference_accuracies) / len(preference_accuracies),
        "sampled_policy_reference_kl_per_token": (
            sampled_kl_token_sum / sampled_kl_token_count if sampled_kl_token_count else None
        ),
        "generated_reward_model_score_mean": sum(reward_scores) / len(reward_scores),
        "generated_reward_model_score_std": (
            float(torch.tensor(reward_scores, dtype=torch.float32).std(unbiased=False))
            if reward_scores else None
        ),
        "generated_response_tokens_mean": sum(response_lengths) / count,
        "generated_response_tokens_min": min(response_lengths),
        "generated_response_tokens_max": max(response_lengths),
        "generated_response_words_mean": sum(response_words) / count,
        "generation_truncated_count": truncated_count,
        "generation_max_new_tokens": max_new_tokens,
        "generation_settings": generation_cfg,
    }

    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / f"{name}_evaluation.json"
    generations_path = results_dir / f"{name}_generations.jsonl"
    summary["generations_file"] = str(generations_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with generations_path.open("w", encoding="utf-8") as f:
        for record in generated_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    clear_gpu(policy, reward_model)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved generated examples to {generations_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--dataset", help="Optional path to custom eval dataset")
    args = ap.parse_args()
    evaluate(args.config, args.adapter, args.name, args.dataset)


if __name__ == "__main__":
    main()
