"""
GRPO (Group Relative Policy Optimization) training with TinyLoRA.

A minimal implementation of GRPO for math reasoning, following the paper:
  "Learning to Reason in 13 Parameters" (2602.04118v1)

Usage (fast iteration, 1.5B):
    accelerate launch --mixed_precision bf16 train_grpo.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --max_gen_len 512 \
        --max_seq_len 1024 \
        --batch_size 16 \
        --micro_batch_size 16 \
        --n_tie 560 \
        --proj_dim 13 \
        --k 2 --compile

Usage (full run, 7B on 2×A100):
    accelerate launch --mixed_precision bf16 train_grpo.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --max_gen_len 512 \
        --max_seq_len 1024 \
        --batch_size 32 \
        --micro_batch_size 32 \
        --n_tie 196 \
        --proj_dim 13 \
        --k 2 --compile --no_gradient_checkpointing

Usage (closer to paper (?) though some parts are a bit unclear tbh, 
            we'll see how it goes, lmk if u try it out! 
            7B on 2×A100):
    accelerate launch train_grpo.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --max_gen_len 4096 \
        --max_seq_len 5120 \
        --batch_size 64 \
        --micro_batch_size 64 \
        --n_tie 196 \
        --proj_dim 13 \
        --no_gradient_checkpointing
"""

import argparse
import re
import random
import os
import contextlib

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed

from tinylora import TinyLoRAModel


# ─── Reward function ────────────────────────────────────────────────────────

def extract_answer(text: str) -> str | None:
    """Extract the final numerical answer from a model response."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return None


def exact_match_reward(prediction: str, ground_truth: str) -> float:
    """Binary reward: 1.0 if the extracted answers match, else 0.0."""
    pred = extract_answer(prediction)
    gt = extract_answer(ground_truth)
    if pred is None or gt is None:
        return 0.0
    try:
        return 1.0 if abs(float(pred) - float(gt)) < 1e-5 else 0.0
    except ValueError:
        return 1.0 if pred.strip() == gt.strip() else 0.0


# ─── GSM8K data helpers ────────────────────────────────────────────────────

class GSM8KDataset(Dataset):
    def __init__(self, split: str = "train", tokenizer=None):
        self.data = []
        ds = load_dataset("openai/gsm8k", "main", split=split)
        for row in ds:
            self.data.append({
                "question": row["question"],
                "answer": row["answer"]
            })
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    return questions, answers


def format_prompt(question: str, tokenizer) -> str:
    """Format a math question into a chat prompt."""
    messages = [
        {"role": "system", "content": "Solve the following math problem step by step. Put your final answer after ####."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ─── GRPO training ──────────────────────────────────────────────────────────

@torch.no_grad()
def generate_completions(
    model, tokenizer, prompts: list[str], k: int, max_gen_len: int, max_seq_len: int, temperature: float = 1.0, device=None
) -> tuple[list[list[str]], list[list[int]]]:
    """Batched generation."""
    tokenizer.padding_side = "left"
    
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=max_seq_len - max_gen_len,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            num_return_sequences=k,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    all_texts = []
    all_lengths = []
    prompt_lens = inputs["input_ids"].shape[1]
    
    for i in range(len(prompts)):
        start_idx = i * k
        end_idx = start_idx + k
        group_texts = []
        group_lens = []
        for j in range(start_idx, end_idx):
            gen_ids = outputs[j, prompt_lens:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            group_texts.append(text)
            group_lens.append(len(gen_ids))
        all_texts.append(group_texts)
        all_lengths.append(group_lens)

    tokenizer.padding_side = "right"
    return all_texts, all_lengths


def compute_grpo_loss_step(
    model,
    tokenizer,
    prompts: list[str],
    completions: list[list[str]],
    rewards: list[list[float]],
    accelerator: Accelerator,
    max_seq_len: int = 1024,
    micro_batch_size: int = 4,
) -> tuple[float, float]:
    """
    Batched GRPO loss computation.
    
    Uses no_sync for ALL micro-batches to avoid DDP deadlocks when
    ranks have different numbers of samples. Gradient sync is done
    manually via all_reduce after this function returns.
    """
    device = accelerator.device
    total_loss = 0.0
    total_samples = 0
    
    all_texts = []
    all_advantages = []
    all_prompt_lens = []
    
    for prompt, comps, rews in zip(prompts, completions, rewards):
        rews_t = torch.tensor(rews, dtype=torch.float32)
        if rews_t.std() < 1e-8:
            continue

        prompt_ids = tokenizer(prompt)["input_ids"]
        p_len = len(prompt_ids)
        advantages = (rews_t - rews_t.mean()) / (rews_t.std() + 1e-8)
        
        for comp, adv in zip(comps, advantages):
            all_texts.append(prompt + comp)
            all_advantages.append(adv)
            all_prompt_lens.append(p_len)

    if not all_texts:
        # No trainable signal on this, 
        # gradients stay at zero, 
        # all_reduce after this function will still work correctly.
        return 0.0, 0

    model.train()
    n_samples = len(all_texts)
    indices = torch.randperm(n_samples).tolist()
    
    # Use no_sync for ALL micro-batches. We manually all_reduce the
    # (tiny) gradients in the training loop after this function returns.
    # This should avoid DDP deadlocks when ranks have different micro-batch counts.
    with accelerator.no_sync(model):
        for i in range(0, n_samples, micro_batch_size):
            batch_indices = indices[i : i + micro_batch_size]
            batch_texts = [all_texts[idx] for idx in batch_indices]
            batch_adv = torch.tensor([all_advantages[idx] for idx in batch_indices], device=device)
            batch_p_lens = torch.tensor([all_prompt_lens[idx] for idx in batch_indices], device=device)
            
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=max_seq_len
            ).to(device)
            
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            
            logits = outputs.logits[:, :-1, :]
            labels = inputs["input_ids"][:, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
            
            mask = inputs["attention_mask"][:, 1:]
            seq_len = labels.size(1)
            range_inds = torch.arange(seq_len, device=device).unsqueeze(0)
            completion_mask = range_inds >= (batch_p_lens.unsqueeze(1) - 1)
            active_mask = mask * completion_mask
            
            seq_log_probs = (token_log_probs * active_mask).sum(dim=-1)
            loss = -(seq_log_probs * batch_adv).mean()
            
            accelerator.backward(loss)
            
            total_loss += loss.item() * len(batch_indices)
            total_samples += len(batch_indices)
            
            del inputs, outputs, logits, loss
    
    return total_loss / max(total_samples, 1), total_samples

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TinyLoRA GRPO Training")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model name or path")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k"])
    parser.add_argument("--rank", type=int, default=2, help="Frozen SVD rank r")
    parser.add_argument("--proj_dim", type=int, default=1, help="Trainable vector dim u")
    parser.add_argument("--n_tie", type=int, default=1,
                        help="Weight tying factor (modules sharing one v). "
                             "Use a large number like 560 for full tying.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--k", type=int, default=4, help="Completions per prompt")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Number of prompts per batch")
    parser.add_argument("--max_gen_len", type=int, default=256,
                        help="Max generation length in tokens")
    parser.add_argument("--max_seq_len", type=int, default=1024,
                        help="Max total sequence length for loss computation "
                             "(prompt + completion, truncated to save VRAM)")
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                        help="Disable gradient checkpointing")
    parser.add_argument("--output_dir", type=str, default="./tinylora_adapter")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--micro_batch_size", type=int, default=4,
                        help="Micro batch size for GRPO loss computation")
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for faster forward/backward")
    args = parser.parse_args()
    
    # Initialize Accelerator
    accelerator = Accelerator(gradient_accumulation_steps=1)
    device = accelerator.device
    
    if args.max_seq_len <= args.max_gen_len:
        raise ValueError(f"max_seq_len ({args.max_seq_len}) must be larger than max_gen_len ({args.max_gen_len})")

    set_seed(args.seed)

    if accelerator.is_main_process:
        print(f"Device: {accelerator.device}")

    # ── Load model & tokenizer ──
    if accelerator.is_main_process:
        print(f"Loading model: {args.model}")
        
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    base_model.to(accelerator.device)

    if not args.no_gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()
        if accelerator.is_main_process:
            print("[TinyLoRA] Gradient checkpointing enabled")

    # ── Wrap with TinyLoRA ──
    model = TinyLoRAModel(
        base_model,
        rank=args.rank,
        proj_dim=args.proj_dim,
        n_tie=args.n_tie,
    )
    
    for p in model.trainable_parameters():
        p.requires_grad = True

    optimizer = AdamW(model.trainable_parameters(), lr=args.lr)

    # Keep a reference to the raw model for generation & saving.
    # torch.compile + DDP are wrappers sharing the same weights.
    raw_model = model
    if args.compile:
        if accelerator.is_main_process:
            print("[TinyLoRA] Compiling model with torch.compile...")
        model = torch.compile(model)

    # ── Load data ──
    if accelerator.is_main_process:
        print("Loading GSM8K...")
    
    dataset = GSM8KDataset("train", tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0, 
        pin_memory=True
    )

    if accelerator.is_main_process:
        print(f"  Train: {len(dataset)} examples")

    # ── Prepare with Accelerator ──
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # ── Training loop ──
    step = 0
    total_steps = len(dataloader) * args.epochs
    
    pbar = tqdm(total=total_steps, disable=not accelerator.is_main_process, desc="Training")

    for epoch in range(args.epochs):
        for batch_questions, batch_answers in dataloader:
            
            prompts = [format_prompt(q, tokenizer) for q in batch_questions]

            # ── Generation (each rank generates independently) ──
            if not args.no_gradient_checkpointing:
                if hasattr(raw_model.model, "gradient_checkpointing_disable"):
                    raw_model.model.gradient_checkpointing_disable()
                raw_model.model.config.use_cache = True

            completions, _ = generate_completions(
                raw_model, tokenizer, prompts, args.k, args.max_gen_len, args.max_seq_len, device=device
            )

            if not args.no_gradient_checkpointing:
                if hasattr(raw_model.model, "gradient_checkpointing_enable"):
                    raw_model.model.gradient_checkpointing_enable()
                raw_model.model.config.use_cache = False

            torch.cuda.empty_cache()

            # ── Rewards ──
            rewards = []
            for comps, answer in zip(completions, batch_answers):
                rews = [exact_match_reward(c, answer) for c in comps]
                rewards.append(rews)

            # ── GRPO update ──
            optimizer.zero_grad()
            
            avg_loss, n_samples = compute_grpo_loss_step(
                model, tokenizer, prompts, completions, rewards,
                accelerator=accelerator,
                max_seq_len=args.max_seq_len,
                micro_batch_size=args.micro_batch_size,
            )
            
            # Manual gradient sync across ranks.
            # We use no_sync inside compute_grpo_loss_step to avoid DDP deadlocks,
            # and all_reduce the tiny gradients here instead
            if torch.distributed.is_initialized():
                for p in raw_model.trainable_parameters():
                    if p.grad is not None:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)
            
            if n_samples > 0:
                num_microbatches = (n_samples + args.micro_batch_size - 1) // args.micro_batch_size
                if num_microbatches > 1:
                    for p in raw_model.trainable_parameters():
                        if p.grad is not None:
                            p.grad.div_(num_microbatches)

            optimizer.step()

            step += 1
            avg_reward = sum(r for rews in rewards for r in rews) / max(sum(len(r) for r in rewards), 1)
            
            if accelerator.is_main_process:
                pbar.update(1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", reward=f"{avg_reward:.3f}")
                if step % args.log_every == 0:
                    tqdm.write(f"[Step {step}] Loss: {avg_loss:.4f}  Avg reward: {avg_reward:.3f}")

    # ── Save ──
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        raw_model.save_adapter(args.output_dir)
        print("Done!")


if __name__ == "__main__":
    main()
