from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.logging_utils import set_seed
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def sequence_logprob(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    response_mask = batch["response_mask"].to(device)

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    log_probs = F.log_softmax(logits.float(), dim=-1)
    target_ids = input_ids[:, 1:]
    shift_log_probs = log_probs[:, :-1, :]
    token_log_probs = torch.gather(
        shift_log_probs,
        dim=-1,
        index=target_ids.unsqueeze(-1),
    ).squeeze(-1)
    token_mask = response_mask[:, 1:]
    return (token_log_probs * token_mask).sum(dim=-1)


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    max_length = int(cfg["max_sequence_length"])
    filtered_rows = []
    skipped = 0
    for row in rows:
        prompt = prompt_messages_from_preference(row)
        try:
            encode_prompt_response(tokenizer, prompt, preference_responses(row)[0], max_length)
            encode_prompt_response(tokenizer, prompt, preference_responses(row)[1], max_length)
            filtered_rows.append(row)
        except ValueError:
            skipped += 1

    if skipped:
        print(f"Filtered {skipped}/{len(rows)} DPO rows because the prompt or response exceeds max_sequence_length={max_length}.")

    if not filtered_rows:
        raise ValueError(
            f"No DPO rows fit max_sequence_length={max_length}; "
            "increase the configured sequence length or inspect the training data."
        )

    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        filtered_rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, max_length),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": filtered_rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    model = bundle["model"]
    loader = bundle["loader"]
    optimizer = bundle["optimizer"]
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    epochs = int(cfg["epochs"])
    grad_accum_steps = int(cfg.get("grad_accum_steps", 1))

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        batches = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (chosen_batch, rejected_batch) in enumerate(loader, start=1):
            with torch.no_grad():
                with reference_mode(model):
                    ref_chosen = sequence_logprob(model, chosen_batch, device)
                    ref_rejected = sequence_logprob(model, rejected_batch, device)

            policy_chosen = sequence_logprob(model, chosen_batch, device)
            policy_rejected = sequence_logprob(model, rejected_batch, device)

            loss, metrics = dpo_loss(
                policy_chosen,
                policy_rejected,
                ref_chosen,
                ref_rejected,
                bundle["beta"],
            )
            loss.backward()
            running_loss += float(loss.detach().item())
            running_acc += float(metrics["preference_accuracy"].item())
            batches += 1

            if step % grad_accum_steps == 0 or step == len(loader):
                accumulated_batches = min(grad_accum_steps, step % grad_accum_steps or grad_accum_steps)
                for parameter in trainable_parameters(model):
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_batches)
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_norm=float(cfg.get("max_grad_norm", 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        epoch_loss = running_loss / max(1, batches)
        epoch_acc = running_acc / max(1, batches)
        print(f"epoch={epoch}/{epochs} loss={epoch_loss:.4f} pref_acc={epoch_acc:.4f}")

    model.save_pretrained(output)
    print(f"Saved adapter to {output}")
    return {"output": str(output), "beta": bundle["beta"], "epochs": epochs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
