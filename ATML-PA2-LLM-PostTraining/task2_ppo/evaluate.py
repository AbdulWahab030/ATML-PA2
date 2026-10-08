"""task2_ppo/evaluate.py — Held-out evaluation for a trained PPO policy adapter.

Entry-point command (run from the repository root)::

    python -m task2_ppo.evaluate --config configs/ppo.yaml \\
        --adapter outputs/task2_ppo/standard --name standard

Computes and saves:
  * Reward model scores (mean, std) on held-out eval prompts.
  * Sampled per-token KL divergence from the frozen reference.
  * Response length statistics (tokens mean/min/max, word count mean).
  * Truncation count.
  * A JSONL file of generated text samples (one per eval prompt).
  * A JSON summary of all numeric metrics.

Cross-platform notes
--------------------------
Device is selected dynamically (CUDA → MPS → CPU).  ``torch.autocast``
is applied to reward-model inference.  ``clear_gpu`` handles both CUDA
and MPS cache clearing.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

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
    """Return the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextlib.contextmanager
def _inference_autocast(device: torch.device):
    """Context manager for mixed-precision inference (no-op on CPU)."""
    if device.type in {"cuda", "mps"}:
        try:
            with torch.autocast(device_type=device.type, enabled=True):
                yield
            return
        except RuntimeError:
            pass
    with contextlib.nullcontext():
        yield


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    config_path: str,
    adapter: str,
    name: str = "standard",
    dataset_path: str | None = None,
) -> dict:
    """Evaluate a trained PPO policy adapter on held-out evaluation prompts.

    Generates one response per eval prompt, scores every response with the
    frozen reward model, and computes sampled KL divergence relative to the
    reference policy embedded in the LoRA adapter.  All results are written
    to ``results/task2_ppo/`` as JSON and JSONL.

    Args:
        config_path:  Path to ``configs/ppo.yaml``.
        adapter:      Path to the LoRA adapter directory to evaluate.
        name:         Run identifier used as a prefix for output file names.
        dataset_path: Optional override for the eval prompt JSONL file.

    Returns:
        Summary dict with all numeric metrics and file-path references.
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
    max_new_tokens = int(cfg.get("eval_max_response_length", 768))
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

    print(f"=== Evaluating PPO Policy: {name} ({adapter}) ===")
    print(f"Evaluating on {len(rows)} prompts with eval_max_response_length={max_new_tokens}…")

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start: start + batch_size]
        prompts = generation_prompts[start: start + batch_size]

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
            sampled_kl_token_sum / sampled_kl_token_count
            if sampled_kl_token_count else None
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
    with generations_path.open("w", encoding="utf-8") as fh:
        for record in generated_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    clear_gpu(policy, reward_model)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved evaluation metrics → {summary_path}")
    print(f"Saved generated samples  → {generations_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry-point for PPO policy evaluation."""
    ap = argparse.ArgumentParser(description="Evaluate a PPO policy adapter on held-out prompts.")
    ap.add_argument("--config", default="configs/ppo.yaml",
                    help="Path to the task PPO config.")
    ap.add_argument("--adapter", required=True,
                    help="Path to the LoRA adapter directory to evaluate.")
    ap.add_argument("--name", default="standard",
                    help="Identifier used in output file names.")
    ap.add_argument("--dataset",
                    help="Optional path to a custom eval-prompt JSONL file.")
    args = ap.parse_args()
    evaluate(args.config, args.adapter, args.name, args.dataset)


if __name__ == "__main__":
    main()
