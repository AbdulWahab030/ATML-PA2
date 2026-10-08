"""task3_grpo/evaluate.py — Held-out evaluation for a trained GRPO policy adapter.

Entry-point command (run from the repository root)::

    python -m task3_grpo.evaluate --config configs/grpo.yaml \
        --adapter outputs/task3_grpo/standard --name standard

Computes and saves:
  * Reward model scores (mean, std) on held-out eval prompts.
  * Sampled per-token KL divergence from the frozen reference.
  * Response length statistics (tokens mean/std/min/max/IQR, word count mean).
  * Truncation count.
  * A JSONL file of generated text samples (one per eval prompt).
  * A JSON summary of all numeric metrics.

Cross-platform notes
--------------------------
Device is dynamically selected (CUDA → MPS → CPU). `clear_gpu` handles both CUDA
and MPS cache clearing gracefully.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import set_seed
from common.metrics import word_count
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode


# ---------------------------------------------------------------------------
# Cross-platform helpers
# ---------------------------------------------------------------------------

def _detect_device() -> torch.device:
    """Return the best available compute device dynamically."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def load_evaluation_bundle(config_path: str, adapter: str) -> dict:
    """Load config, evaluation prompts, models, and tokenizer."""
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate(
    config_path: str,
    adapter: str,
    name: str = "standard",
    dataset_path: str | None = None,
) -> dict:
    """Evaluate a trained GRPO policy adapter on held-out prompts.

    Args:
        config_path:  Path to `configs/grpo.yaml`.
        adapter:      Path to the LoRA adapter directory.
        name:         Run name prefix for saving results.
        dataset_path: Optional override for the evaluation dataset path.

    Returns:
        Summary dict of evaluation metrics.
    """
    cfg = load_yaml(config_path)
    set_seed(int(cfg.get("seed", 6304)))

    device = _detect_device()
    print(f"[device] Using: {device}")

    eval_file = dataset_path or cfg["paths"]["rl_prompt_eval"]
    rows = read_jsonl(eval_file)
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    policy.to(device)
    reward_model, reward_tokenizer = load_reward_model(cfg)

    max_prompt_length = int(cfg.get("max_prompt_length", 256))
    max_new_tokens = int(cfg.get("cache_generation_cap", 768))
    reward_max_length = int(cfg.get("reward_max_length", 1280))
    batch_size = int(cfg.get("eval_batch_size", 2))
    generation_cfg = cfg.get("generation", {})

    generation_prompts = [prompt_messages(row) for row in rows]
    generated_records: list[dict] = []
    sampled_kl_token_sum = 0.0
    sampled_kl_token_count = 0
    reward_scores: list[float] = []
    response_lengths: list[int] = []
    response_words: list[int] = []
    truncated_count = 0

    print(f"=== Evaluating GRPO Policy: {name} ({adapter}) ===")
    print(f"Evaluating on {len(rows)} prompts with max_new_tokens={max_new_tokens}...")

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        prompts = generation_prompts[start : start + batch_size]

        with _inference_autocast(device):
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

        with torch.no_grad():
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

        with _inference_autocast(device):
            scores = score_reward_pairs(
                reward_model,
                reward_tokenizer,
                prompts,
                generated["responses"],
                max_length=reward_max_length,
            )
        batch_scores = [float(s) for s in scores.detach().cpu().tolist()]
        reward_scores.extend(batch_scores)
        response_lengths.extend(int(n) for n in generated["response_lengths"])
        response_words.extend(word_count(t) for t in generated["responses"])
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

    clear_gpu()

    count = len(rows)
    lengths_arr = np.array(response_lengths, dtype=float) if response_lengths else np.array([0.0])
    q75, q25 = np.percentile(lengths_arr, [75, 25])
    iqr_len = float(q75 - q25)

    summary = {
        "name": name,
        "adapter": str(adapter),
        "evaluation_dataset": str(eval_file),
        "num_prompts_evaluated": count,
        "reward_score_mean": float(np.mean(reward_scores)) if reward_scores else 0.0,
        "reward_score_std": float(np.std(reward_scores)) if reward_scores else 0.0,
        "sampled_policy_reference_kl_per_token": (
            sampled_kl_token_sum / sampled_kl_token_count if sampled_kl_token_count else 0.0
        ),
        "response_length_tokens_mean": float(np.mean(lengths_arr)),
        "response_length_tokens_std": float(np.std(lengths_arr)),
        "response_length_tokens_iqr": iqr_len,
        "response_length_tokens_min": int(np.min(lengths_arr)),
        "response_length_tokens_max": int(np.max(lengths_arr)),
        "response_words_mean": float(np.mean(response_words)) if response_words else 0.0,
        "truncated_sequences_count": truncated_count,
        "truncated_sequences_fraction": truncated_count / max(1, count),
    }

    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_file = results_dir / f"{name}_eval.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    gen_file = results_dir / f"{name}_eval_generations.jsonl"
    with gen_file.open("w", encoding="utf-8") as f:
        for rec in generated_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n[Evaluation Complete] {name}")
    print(f"  Mean Reward : {summary['reward_score_mean']:.4f} ± {summary['reward_score_std']:.4f}")
    print(f"  Sampled KL  : {summary['sampled_policy_reference_kl_per_token']:.5f}")
    print(f"  Mean Tokens : {summary['response_length_tokens_mean']:.1f} (IQR: {iqr_len:.1f})")
    print(f"  Truncated   : {truncated_count}/{count} ({summary['truncated_sequences_fraction']:.1%})")
    print(f"  Saved to    : {summary_file}")

    return summary


def main():
    ap = argparse.ArgumentParser(description="Evaluate a GRPO policy adapter.")
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True, help="Path to policy adapter directory")
    ap.add_argument("--name", default="standard", help="Identifier prefix for results")
    ap.add_argument("--dataset", default=None, help="Optional override for evaluation prompt JSONL")
    args = ap.parse_args()

    evaluate(args.config, args.adapter, args.name, args.dataset)


if __name__ == "__main__":
    main()
