from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import batch_generate
from common.logging_utils import set_seed
from common.metrics import parse_word_limit, word_count, word_limit_compliance
from common.models import clear_gpu, load_policy, load_tokenizer, reference_mode
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import run_training


def _sequence_logprob(model, batch, device):
    """Compute unnormalized sequence log-probability over response tokens."""
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


def _eval_stratified_dpo(model, tokenizer, rows: list[dict], cfg: dict, device):
    """Evaluate preference accuracy, margins, and generation length per stratum."""
    max_length = int(cfg["max_sequence_length"])
    batch_size = int(cfg.get("eval_batch_size", 2))
    beta = float(cfg.get("beta", 0.10))

    # 1. Held-out preference scoring
    stratum_records = defaultdict(list)
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        chosen_list, rejected_list = [], []
        valid_indices = []

        for idx, row in enumerate(batch_rows):
            prompt = prompt_messages_from_preference(row)
            c_text, r_text = preference_responses(row)
            try:
                chosen_list.append(encode_prompt_response(tokenizer, prompt, c_text, max_length))
                rejected_list.append(encode_prompt_response(tokenizer, prompt, r_text, max_length))
                valid_indices.append(idx)
            except ValueError:
                continue

        if not valid_indices:
            continue

        chosen_batch = pad_batch(tokenizer, chosen_list)
        rejected_batch = pad_batch(tokenizer, rejected_list)

        with torch.inference_mode():
            policy_chosen = _sequence_logprob(model, chosen_batch, device)
            policy_rejected = _sequence_logprob(model, rejected_batch, device)
            with reference_mode(model):
                ref_chosen = _sequence_logprob(model, chosen_batch, device)
                ref_rejected = _sequence_logprob(model, rejected_batch, device)

            policy_margin = policy_chosen - policy_rejected
            ref_margin = ref_chosen - ref_rejected
            pref_margin = policy_margin - ref_margin
            loss, _ = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)

        for i, idx in enumerate(valid_indices):
            row = batch_rows[idx]
            stratum = row.get("length_stratum", "unknown")
            is_correct = bool(pref_margin[i].item() > 0)
            margin_val = float(pref_margin[i].item())
            loss_val = float(loss.item())

            stratum_records[stratum].append({
                "prompt_id": row.get("prompt_id"),
                "is_correct": is_correct,
                "margin": margin_val,
                "loss": loss_val,
                "length_difference": row.get("length_difference", 0),
            })

    # 2. Generation length statistics per stratum
    generation_cfg = cfg.get("generation", {})
    max_new_tokens = int(cfg.get("max_generation_tokens", 256))
    max_prompt_length = int(cfg.get("max_prompt_length", max_length))

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        prompts = [prompt_messages_from_preference(r) for r in batch_rows]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            temperature=float(generation_cfg.get("temperature", 0.7)),
            top_p=float(generation_cfg.get("top_p", 0.9)),
            do_sample=bool(generation_cfg.get("do_sample", True)),
        )
        for row, text, tokens in zip(batch_rows, gen["responses"], gen["response_lengths"]):
            stratum = row.get("length_stratum", "unknown")
            stratum_records[f"{stratum}_lengths"].append({
                "tokens": int(tokens),
                "words": word_count(text),
            })

    # Compute aggregate summary per stratum
    summary = {}
    for stratum in ["preferred_longer", "length_matched", "rejected_longer"]:
        recs = stratum_records.get(stratum, [])
        lens = stratum_records.get(f"{stratum}_lengths", [])
        if recs:
            acc = sum(r["is_correct"] for r in recs) / len(recs)
            mean_margin = sum(r["margin"] for r in recs) / len(recs)
            mean_loss = sum(r["loss"] for r in recs) / len(recs)
        else:
            acc, mean_margin, mean_loss = 0.0, 0.0, 0.0

        mean_tokens = sum(l["tokens"] for l in lens) / len(lens) if lens else 0.0
        mean_words = sum(l["words"] for l in lens) / len(lens) if lens else 0.0

        summary[stratum] = {
            "num_examples": len(recs),
            "preference_accuracy": acc,
            "mean_margin": mean_margin,
            "mean_loss": mean_loss,
            "generated_tokens_mean": mean_tokens,
            "generated_words_mean": mean_words,
        }

    return summary


def _eval_word_limits(model, tokenizer, word_prompts: list[dict], cfg: dict):
    """Evaluate instruction-following and explicit word-limit compliance."""
    generation_cfg = cfg.get("generation", {})
    max_new_tokens = int(cfg.get("max_generation_tokens", 256))
    prompts = [prompt_messages(r) for r in word_prompts]

    gen = batch_generate(
        model,
        tokenizer,
        prompts,
        max_prompt_length=int(cfg.get("max_sequence_length", 768)),
        max_new_tokens=max_new_tokens,
        temperature=float(generation_cfg.get("temperature", 0.7)),
        top_p=float(generation_cfg.get("top_p", 0.9)),
        do_sample=False,  # Deterministic generation for instruction compliance
    )

    results = []
    compliant_count = 0
    total_valid = 0

    for row, response in zip(word_prompts, gen["responses"]):
        p_text = str(row.get("prompt", row.get("messages", [{}])[0].get("content", "")))
        limit = parse_word_limit(p_text)
        w_count = word_count(response)
        compliance = word_limit_compliance(p_text, response)

        if compliance is not None:
            total_valid += 1
            if compliance == 1.0:
                compliant_count += 1

        results.append({
            "prompt_id": row.get("prompt_id"),
            "prompt": p_text,
            "limit": limit,
            "word_count": w_count,
            "compliant": bool(compliance == 1.0) if compliance is not None else None,
            "response_sample": response[:120] + ("..." if len(response) > 120 else ""),
        })

    compliance_rate = compliant_count / max(1, total_valid)
    mean_words = sum(r["word_count"] for r in results) / max(1, len(results))

    return {
        "compliance_rate": compliance_rate,
        "mean_words": mean_words,
        "total_prompts": len(results),
        "details": results,
    }


def analyze_length(
    config_path: str = "configs/dpo.yaml",
    standard_adapter: str | None = None,
    length_adapter: str | None = None,
    skip_train: bool = False,
):
    """Train length-balanced DPO and compare with standard DPO across length strata and word limits."""
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg["base_model"])

    std_path = repo_path(standard_adapter or cfg.get("standard_output", "outputs/task1_dpo/standard"))
    len_path = repo_path(length_adapter or cfg.get("length_output", "outputs/task1_dpo/length_balanced"))
    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train length-balanced DPO if needed
    if not skip_train:
        print("\n=== Training Length-Balanced DPO Model ===")
        run_training(
            config_path=config_path,
            run_name="length_balanced",
            dataset_path=cfg["paths"]["dpo_length_train"],
            output_path=str(len_path),
            beta=float(cfg["beta"]),
        )
        clear_gpu()
    else:
        print(f"\nSkipping training; using existing adapter at {len_path}")

    stratified_rows = read_jsonl(cfg["paths"]["dpo_length_eval"])
    word_prompts = read_jsonl(cfg["paths"]["word_limit_prompts"])

    models_to_test = [
        ("standard", std_path),
        ("length_balanced", len_path),
    ]

    analysis_results = {}

    for name, adapter_path in models_to_test:
        print(f"\n=== Evaluating Model: {name} ({adapter_path}) ===")
        if not adapter_path.exists():
            print(f"Warning: Adapter {adapter_path} not found. Skipping evaluation for {name}.")
            continue

        model = load_policy(cfg, adapter_path=str(adapter_path), trainable=False)
        model.to(device)

        print(f"Evaluating on length-stratified dataset ({len(stratified_rows)} rows)...")
        strata_summary = _eval_stratified_dpo(model, tokenizer, stratified_rows, cfg, device)

        print(f"Evaluating word-limit compliance ({len(word_prompts)} prompts)...")
        word_summary = _eval_word_limits(model, tokenizer, word_prompts, cfg)

        analysis_results[name] = {
            "strata": strata_summary,
            "word_limits": word_summary,
        }

        clear_gpu(model)

    # Save summary JSON
    summary_path = results_dir / "length_analysis_summary.json"
    summary_path.write_text(json.dumps(analysis_results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Build and save comparison table CSV for strata
    table_rows = []
    for model_name, res in analysis_results.items():
        for stratum, metrics in res["strata"].items():
            table_rows.append({
                "model": model_name,
                "stratum": stratum,
                "preference_accuracy": metrics["preference_accuracy"],
                "mean_margin": metrics["mean_margin"],
                "generated_tokens_mean": metrics["generated_tokens_mean"],
                "generated_words_mean": metrics["generated_words_mean"],
            })
    strata_df = pd.DataFrame(table_rows)
    strata_csv = results_dir / "length_strata_comparison.csv"
    strata_df.to_csv(strata_csv, index=False)

    # Build and save word limit comparison CSV
    wl_rows = []
    for model_name, res in analysis_results.items():
        wl_info = res["word_limits"]
        wl_rows.append({
            "model": model_name,
            "compliance_rate": wl_info["compliance_rate"],
            "mean_words": wl_info["mean_words"],
        })
    wl_df = pd.DataFrame(wl_rows)
    wl_csv = results_dir / "word_limit_summary.csv"
    wl_df.to_csv(wl_csv, index=False)

    print("\n=== Length-Confounding Analysis Complete ===")
    print("\n--- Stratum Breakdown ---")
    print(strata_df.to_string(index=False))
    print("\n--- Word-Limit Compliance ---")
    print(wl_df.to_string(index=False))
    print(f"\nArtifacts saved to {summary_path}, {strata_csv}, and {wl_csv}")
    return analysis_results


def main():
    ap = argparse.ArgumentParser(description="Run DPO length confounding and instruction compliance study.")
    ap.add_argument("--config", default="configs/dpo.yaml", help="Path to DPO config file.")
    ap.add_argument("--standard-adapter", help="Path to trained standard DPO adapter.")
    ap.add_argument("--length-adapter", help="Path to trained length-balanced DPO adapter.")
    ap.add_argument("--skip-train", action="store_true", help="Skip training length-balanced model if adapter exists.")
    args = ap.parse_args()

    analyze_length(args.config, args.standard_adapter, args.length_adapter, args.skip_train)


if __name__ == "__main__":
    main()
