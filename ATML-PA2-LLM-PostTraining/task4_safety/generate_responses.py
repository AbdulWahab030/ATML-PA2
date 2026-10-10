from __future__ import annotations

import argparse
import contextlib
import gc
from pathlib import Path

import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.models import clear_gpu, load_policy, load_tokenizer


def get_device() -> torch.device:
    """Dynamically select device: CUDA -> MPS -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextlib.contextmanager
def inference_autocast(device: torch.device):
    """Graceful mixed-precision context for inference (CUDA / MPS / fallback CPU)."""
    if device.type in {"cuda", "mps"}:
        try:
            with torch.autocast(device_type=device.type, enabled=True):
                yield
            return
        except RuntimeError:
            pass
    with contextlib.nullcontext():
        yield


def safe_empty_cache():
    """Safely clear accelerator cache on CUDA or MPS."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def policy_specs(cfg: dict) -> dict[str, str | None]:
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(
    cfg: dict,
    policy_name: str,
    batch_size: int = 4,
    device: torch.device | None = None,
    out_file: Path | None = None,
    force: bool = False,
) -> list[dict]:
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(f"Unknown policy '{policy_name}'. Valid policies: {list(specs.keys())}")

    df = load_xstest(cfg)

    # Check for existing complete generation if caching
    if out_file and out_file.exists() and not force:
        try:
            existing = read_jsonl(out_file)
            if len(existing) == len(df):
                print(f"[{policy_name}] Already generated ({len(existing)} rows in {out_file}). Skipping.")
                return existing
        except Exception:
            pass

    adapter = specs[policy_name]
    if adapter is not None:
        full_adapter = repo_path(adapter)
        if not full_adapter.exists():
            raise FileNotFoundError(
                f"Adapter checkpoint for policy '{policy_name}' not found at {full_adapter}."
            )

    if device is None:
        device = get_device()
    print(f"[{policy_name}] Loading policy on device: {device} (adapter: {adapter})")

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)

    max_new_tokens = int(cfg.get("safety_max_new_tokens", 256))
    records: list[dict] = []
    total_prompts = len(df)

    print(f"[{policy_name}] Generating deterministic responses for {total_prompts} XSTest prompts (batch_size={batch_size}, max_new_tokens={max_new_tokens})...")

    try:
        from tqdm import tqdm
        pbar = tqdm(range(0, total_prompts, batch_size), desc=f"Gen {policy_name}")
    except ImportError:
        pbar = range(0, total_prompts, batch_size)

    try:
        for start in pbar:
            chunk = df.iloc[start : start + batch_size]
            prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]

            with inference_autocast(device):
                gen = batch_generate(
                    model,
                    tokenizer,
                    prompts,
                    max_prompt_length=256,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False,
                )

            for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
                records.append({
                    "xstest_id": int(row["xstest_id"]),
                    "id": str(row.get("id", "")),
                    "policy": policy_name,
                    "prompt": str(row["prompt"]),
                    "benchmark_class": str(row["benchmark_class"]),
                    "type": str(row["type"]),
                    "label": str(row.get("label", "")),
                    "focus": str(row.get("focus", "")),
                    "note": str(row.get("note", "")),
                    "response": response,
                    "response_tokens": int(n_tok),
                })
    finally:
        clear_gpu(model, tokenizer)
        safe_empty_cache()

    # Preserve strictly fixed xstest_id order
    records.sort(key=lambda r: r["xstest_id"])

    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(out_file, records)
        print(f"[{policy_name}] Wrote {len(records)} responses to {out_file}")

    return records


def main():
    ap = argparse.ArgumentParser(description="Task 4, Step 1: Generate deterministic responses on XSTest.")
    ap.add_argument("--config", default="configs/feedback.yaml", help="Path to config file.")
    ap.add_argument(
        "--policy",
        choices=["sft", "dpo", "ppo", "grpo", "all"],
        default="all",
        help="Policy to generate responses for (default: all).",
    )
    ap.add_argument("--batch-size", type=int, default=4, help="Batch size for generation.")
    ap.add_argument("--force", action="store_true", help="Force regeneration even if output exists.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = get_device()
    print(f"Active compute device: {device}")

    outdir = repo_path(cfg.get("results_dir", "results")) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)

    policies_to_run = (
        ["sft", "dpo", "ppo", "grpo"]
        if args.policy == "all"
        else [args.policy]
    )

    for pol in policies_to_run:
        out_file = outdir / f"generated_{pol}.jsonl"
        try:
            generate_for_policy(
                cfg,
                policy_name=pol,
                batch_size=args.batch_size,
                device=device,
                out_file=out_file,
                force=args.force,
            )
        except FileNotFoundError as e:
            print(f"[WARNING] Skipping policy '{pol}': {e}")
        except Exception as e:
            print(f"[ERROR] Failed generation for policy '{pol}': {e}")
            if args.policy != "all":
                raise


if __name__ == "__main__":
    main()
