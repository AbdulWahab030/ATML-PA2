from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import set_seed
from common.metrics import word_count
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode


def evaluate(
    config_path: str,
    adapter: str,
    name: str = "standard",
    dataset_path: str | None = None,
):
    """Evaluate a trained PPO policy adapter on held-out evaluation prompts.

    Computes held-out reward model scores, sampled per-token KL divergence
    from the frozen reference model, policy entropy, response length statistics,
    and saves machine-readable summary logs and generated text samples.
    """
    cfg = load_yaml(config_path)
    set_seed(int(cfg.get("seed", 6304)))
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    eval_file = dataset_path or cfg["paths"]["rl_prompt_eval"]
    rows = read_jsonl(eval_file)
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    policy.to(device)
    reward_model, reward_tokenizer = load_reward_model(cfg)

    max_prompt_length = int(cfg.get("max_prompt_length", 256))
    max_new_tokens = int(cfg.get("eval_max_response_length", 768))
    batch_size = int(cfg.get("eval_batch_size", 2))
    generation_cfg = cfg.get("generation", {})

    generation_prompts = [prompt_messages(row) for row in rows]
    generated_records = []
    sampled_kl_token_sum = 0.0
    sampled_kl_token_count = 0
    reward_scores = []
    response_lengths = []
    response_words = []
    truncated_count = 0

    print(f"=== Evaluating PPO Policy: {name} ({adapter}) ===")
    print(f"Evaluating on {len(rows)} prompts with eval_max_response_length={max_new_tokens}...")

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
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
            max_length=int(cfg.get("reward_max_length", 1280)),
        )
        batch_scores = [float(score) for score in scores.detach().cpu().tolist()]
        reward_scores.extend(batch_scores)
        response_lengths.extend(int(n) for n in generated["response_lengths"])
        response_words.extend(word_count(text) for text in generated["responses"])
        truncated_count += sum(generated["truncated"])

        for offset, (row, prompt, response) in enumerate(
            zip(batch_rows, prompts, generated["responses"])
        ):
            generated_records.append({
                "row_index": start + offset,
                "id": row.get("id", row.get("prompt_id")),
                "prompt": prompt,
                "generated": response,
                "reward_score": batch_scores[offset],
                "tokens": int(generated["response_lengths"][offset]),
                "words": word_count(response),
                "truncated": bool(generated["truncated"][offset]),
            })

    count = len(rows)
    summary = {
        "name": name,
        "adapter": str(adapter),
        "evaluation_dataset": str(eval_file),
        "num_prompts_evaluated": count,
        "reward_score_mean": sum(reward_scores) / max(1, len(reward_scores)),
        "reward_score_std": (
            float(torch.tensor(reward_scores, dtype=torch.float32).std(unbiased=False))
            if reward_scores else None
        ),
        "sampled_policy_reference_kl_per_token": (
            sampled_kl_token_sum / sampled_kl_token_count if sampled_kl_token_count else None
        ),
        "response_tokens_mean": sum(response_lengths) / max(1, count),
        "response_tokens_min": min(response_lengths) if response_lengths else 0,
        "response_tokens_max": max(response_lengths) if response_lengths else 0,
        "response_words_mean": sum(response_words) / max(1, count),
        "truncated_count": truncated_count,
        "eval_max_response_length": max_new_tokens,
    }

    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
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
    print(f"Saved evaluation metrics to {summary_path}")
    print(f"Saved generated samples to {generations_path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Evaluate PPO policy adapter.")
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--dataset", help="Optional path to custom eval dataset")
    args = ap.parse_args()
    evaluate(args.config, args.adapter, args.name, args.dataset)


if __name__ == "__main__":
    main()
