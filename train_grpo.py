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
        --k 2

Usage (full run, 7B on 2x RTX PRO 6000):
    accelerate launch --mixed_precision bf16 train_grpo.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --max_gen_len 512 \
        --max_seq_len 1024 \
        --batch_size 32 \
        --micro_batch_size 32 \
        --n_tie 196 \
        --proj_dim 13 \
        --k 2 --no_gradient_checkpointing

Usage (with vLLM, single GPU):
    accelerate launch --num_processes=1 --mixed_precision bf16 train_grpo.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --max_gen_len 512 \
        --max_seq_len 1024 \
        --batch_size 32 \
        --micro_batch_size 32 \
        --n_tie 196 \
        --proj_dim 13 \
        --k 2 --no_gradient_checkpointing \
        --use_vllm --vllm_gpu_ratio 0.4

Usage (with vLLM, 2 GPUs, dedicated split: training on GPU 0, vLLM on GPU 1):
    accelerate launch --num_processes=1 --mixed_precision bf16 train_grpo.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --max_gen_len 512 \
        --max_seq_len 1024 \
        --batch_size 32 \
        --micro_batch_size 32 \
        --n_tie 196 \
        --proj_dim 13 \
        --k 2 --no_gradient_checkpointing \
        --use_vllm --vllm_gpu_id 1 --vllm_gpu_ratio 0.9

Usage (closer to paper (?) though some parts are a bit unclear tbh,
            we'll see how it goes, lmk if u try it out!
            7B on 2x RTX PRO 6000, use --use_vllm for the long seqlen):
    accelerate launch --num_processes=1 --mixed_precision bf16 train_grpo.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --max_gen_len 4096 \
        --max_seq_len 5120 \
        --batch_size 64 \
        --micro_batch_size 64 \
        --n_tie 196 \
        --proj_dim 13 \
        --no_gradient_checkpointing \
        --use_vllm --vllm_tp_size 2 --vllm_gpu_ratio 0.4
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
    # Try #### pattern (GSM8K style)
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    # Try \boxed{} pattern (MATH style)
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    # Fallback: last number in text
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
        try:
            ds = load_dataset("openai/gsm8k", "main", split=split)
            for row in ds:
                self.data.append({"question": row["question"], "answer": row["answer"]})
        except Exception as e:
            print(f"Error loading dataset: {e}")
            # Fallback for testing if dataset fails to load (e.g. no internet)
            self.data = [{"question": "1+1", "answer": "2"}] * 10
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
        {
            "role": "system",
            "content": "Solve the following math problem step by step. Put your final answer after ####.",
        },
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ─── HuggingFace generation (fallback when vLLM is not used) ────────────────


@torch.no_grad()
def generate_completions_hf(
    model,
    tokenizer,
    prompts: list[str],
    k: int,
    max_gen_len: int,
    max_seq_len: int,
    temperature: float = 1.0,
    device=None,
) -> tuple[list[list[str]], list[list[int]]]:
    """
    Batched generation using HuggingFace generate(). fast af.
    """
    # Set padding to LEFT for generation (for CausalLM batching)
    tokenizer.padding_side = "left"

    # Tokenize all prompts at once
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len - max_gen_len,
    ).to(device)

    # Generate all k completions in one go (or parallel streams)
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

    # outputs shape: [batch_size * k, total_len]
    # We need to split them back into groups of k

    all_texts = []
    all_lengths = []

    # The outputs are ordered: [p1_k1, p1_k2, ..., p2_k1, p2_k2, ...]
    prompt_lens = inputs["input_ids"].shape[1]

    for i in range(len(prompts)):
        start_idx = i * k
        end_idx = start_idx + k

        group_texts = []
        group_lens = []

        for j in range(start_idx, end_idx):
            # Extract only the new tokens
            gen_ids = outputs[j, prompt_lens:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            group_texts.append(text)
            group_lens.append(len(gen_ids))

        all_texts.append(group_texts)
        all_lengths.append(group_lens)

    # Reset padding side just in case
    tokenizer.padding_side = "right"

    return all_texts, all_lengths


# ─── vLLM generation ────────────────────────────────────────────────────────


# DDP env vars that leak into vLLM's spawned subprocess and cause NCCL deadlocks.
# We scrub them before vLLM init and restore after.
_DDP_ENV_VARS = [
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID",
    "NCCL_ASYNC_ERROR_HANDLING",
]


def init_vllm_engine(
    model_name: str,
    gpu_memory_utilization: float = 0.4,
    max_model_len: int = 4096,
    enforce_eager: bool = True,
    tp_size: int = 1,
    gpu_id: int | None = None,
):
    """
    Initialize a vLLM LLM engine for inference.

    Args:
        tp_size: Tensor parallel size. Use >1 to shard the model across
                 multiple GPUs for faster generation.
        gpu_id:  If set, pin vLLM to this specific GPU (for single-GPU vLLM
                 alongside a training model on another GPU). Ignored if tp_size > 1.
    """
    from vllm import LLM

    # Scrub DDP env vars to prevent EngineCore subprocess conflicts
    saved_env = {}
    for var in _DDP_ENV_VARS:
        if var in os.environ:
            saved_env[var] = os.environ.pop(var)
            print(f"[vLLM] Scrubbed env var {var}={saved_env[var]}")

    # Pin vLLM to a specific GPU if requested (only for tp_size=1)
    saved_cuda_visible = None
    if gpu_id is not None and tp_size == 1:
        saved_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[vLLM] Pinning to GPU {gpu_id} via CUDA_VISIBLE_DEVICES")

    print(f"[vLLM] Creating LLM engine (tp_size={tp_size})...")
    try:
        llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            dtype="bfloat16",
            tensor_parallel_size=tp_size,
            disable_custom_all_reduce=True,
        )
        print("[vLLM] LLM engine created successfully")
    finally:
        # Restore env vars
        for var, val in saved_env.items():
            os.environ[var] = val
        if saved_cuda_visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved_cuda_visible
        elif gpu_id is not None and tp_size == 1:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    return llm


# NOTE: Weight hot-swapping was removed. vLLM V1 (0.17+) runs the model
# in a separate EngineCore subprocess with no reliable public API for
# in-memory weight updates. Instead, we initialize vLLM once with the
# base model weights and rely on truncated importance sampling to correct
# for the policy mismatch. TinyLoRA's SVD structure constrains each
# adapted layer to a small rank-1 perturbation, keeping the delta between
# the base model and the LoRA model small regardless of config.


@torch.no_grad()
def generate_completions_vllm(
    llm,
    prompts: list[str],
    k: int,
    max_gen_len: int,
    temperature: float = 1.0,
) -> tuple[list[list[str]], list[list[int]]]:
    """
    Batched generation using vLLM. Each prompt gets k completions.

    Returns:
        all_texts: list[list[str]] — [n_prompts, k] completions
        all_lengths: list[list[int]] — [n_prompts, k] token lengths
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        n=k,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_gen_len,
    )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    all_texts = []
    all_lengths = []
    for output in outputs:
        group_texts = []
        group_lens = []
        for completion in output.outputs:
            group_texts.append(completion.text)
            group_lens.append(len(completion.token_ids))
        all_texts.append(group_texts)
        all_lengths.append(group_lens)

    return all_texts, all_lengths


# ─── GRPO training ──────────────────────────────────────────────────────────


@torch.no_grad()
def compute_behavior_log_probs(
    model,
    tokenizer,
    prompts: list[str],
    completions: list[list[str]],
    max_seq_len: int,
    device,
) -> list[list[float]]:
    """
    Compute per-sequence log-probs under the MERGED (behavior) policy.

    Used for importance sampling correction: we need log π_merged(y|x)
    for each completion y generated by the merged model.

    This runs the full forward pass through the TinyLoRA model with merged
    weights to get the behavior policy log-probs.
    """
    model.eval()
    all_log_probs = []

    for prompt, comps in zip(prompts, completions):
        prompt_ids = tokenizer(prompt)["input_ids"]
        p_len = len(prompt_ids)

        batch_texts = [prompt + c for c in comps]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
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
        completion_mask = range_inds >= (p_len - 1)
        active_mask = mask * completion_mask

        seq_log_probs = (token_log_probs * active_mask).sum(dim=-1)
        all_log_probs.append(seq_log_probs.tolist())

        del inputs, outputs, logits

    return all_log_probs


def compute_grpo_loss_step(
    model,
    tokenizer,
    prompts: list[str],
    completions: list[list[str]],
    rewards: list[list[float]],
    accelerator: Accelerator,
    max_seq_len: int = 1024,
    micro_batch_size: int = 4,
    importance_weights: list[list[float]] | None = None,
) -> tuple[float, float]:
    """
    Batched GRPO loss computation.
    Significantly faster by parallelizing the forward/backward passes.

    When importance_weights is provided (vLLM mode), each sample's advantage
    is multiplied by its truncated importance weight to correct for the
    policy mismatch between the merged model (behavior) and the true LoRA
    model (target).
    """
    device = accelerator.device
    total_loss = 0.0
    total_samples = 0
    rank = accelerator.process_index

    all_texts = []
    all_advantages = []
    all_prompt_lens = []

    for idx, (prompt, comps, rews) in enumerate(zip(prompts, completions, rewards)):
        rews_t = torch.tensor(rews, dtype=torch.float32)
        if rews_t.std() < 1e-8:
            continue  # Skip if no signal

        # Compute prompt length for masking
        # NOTE: We re-tokenize the prompt to get the exact length
        # This is fast enough since batch size is small
        prompt_ids = tokenizer(prompt)["input_ids"]
        p_len = len(prompt_ids)

        # Standard GRPO advantage (group norm)
        advantages = (rews_t - rews_t.mean()) / (rews_t.std() + 1e-8)

        # Apply truncated importance weights if provided (vLLM mode)
        if importance_weights is not None:
            iw = torch.tensor(importance_weights[idx], dtype=torch.float32)
            advantages = advantages * iw

        for comp, adv in zip(comps, advantages):
            all_texts.append(prompt + comp)
            all_advantages.append(adv)
            all_prompt_lens.append(p_len)

    if not all_texts:
        return 0.0, 0

    # (Since we are doing training, activations consume memory)
    model.train()

    n_samples = len(all_texts)
    # Shuffle to mix advantages (optional but good for stability)
    indices = torch.randperm(n_samples).tolist()

    # Use no_sync for all micro-batches except the last one.
    # This avoids redundant DDP all-reduce on every backward() call.
    micro_batch_starts = list(range(0, n_samples, micro_batch_size))

    for mb_idx, i in enumerate(micro_batch_starts):
        is_last_microbatch = mb_idx == len(micro_batch_starts) - 1
        # Skip gradient sync for all but the last micro-batch
        ctx = (
            contextlib.nullcontext()
            if is_last_microbatch
            else accelerator.no_sync(model)
        )
        batch_indices = indices[i : i + micro_batch_size]
        batch_texts = [all_texts[idx] for idx in batch_indices]
        batch_adv = torch.tensor(
            [all_advantages[idx] for idx in batch_indices], device=device
        )
        batch_p_lens = torch.tensor(
            [all_prompt_lens[idx] for idx in batch_indices], device=device
        )

        # Tokenize the micro-batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(device)

        # GRPO requires computing log_prob(completion | prompt)
        # We mask out the prompt tokens from the loss

        with ctx:
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            # Shift for autoregressive loss
            logits = outputs.logits[:, :-1, :]
            labels = inputs["input_ids"][:, 1:]

            # Compute token-level log probs
            log_probs = F.log_softmax(logits, dim=-1)

            # Gather log probs of the actual tokens
            # shape: [B, L]
            token_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(
                -1
            )

            # Mask padding in labels
            mask = inputs["attention_mask"][:, 1:]

            # Mask prompt tokens (we only want to train on the completion)
            seq_len = labels.size(1)
            range_inds = torch.arange(seq_len, device=device).unsqueeze(0)
            completion_mask = range_inds >= (batch_p_lens.unsqueeze(1) - 1)

            # Apply both padding mask and completion mask
            active_mask = mask * completion_mask

            # Sum log_probs per sequence (masked)
            seq_log_probs = (token_log_probs * active_mask).sum(dim=-1)

            # GRPO Loss: -1 * (log_prob * advantage)
            loss = -(seq_log_probs * batch_adv).mean()

            # Backprop with accelerator (sync only on last micro-batch)
            accelerator.backward(loss)

        total_loss += loss.item() * len(batch_indices)
        total_samples += len(batch_indices)

        del inputs, outputs, logits, loss

    return total_loss / max(total_samples, 1), total_samples


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="TinyLoRA GRPO Training")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k"])
    parser.add_argument("--rank", type=int, default=2, help="Frozen SVD rank r")
    parser.add_argument(
        "--proj_dim", type=int, default=1, help="Trainable vector dim u"
    )
    parser.add_argument(
        "--n_tie",
        type=int,
        default=1,
        help="Weight tying factor (modules sharing one v). "
        "Use a large number like 560 for full tying.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--k", type=int, default=4, help="Completions per prompt")
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Number of prompts per batch"
    )
    parser.add_argument(
        "--max_gen_len", type=int, default=256, help="Max generation length in tokens"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Max total sequence length for loss computation "
        "(prompt + completion, truncated to save VRAM)",
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="Disable gradient checkpointing",
    )
    parser.add_argument("--output_dir", type=str, default="./tinylora_adapter")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=4,
        help="Micro batch size for GRPO loss computation",
    )
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="tinylora-grpo",
        help="W&B project name (requires --wandb)",
    )
    parser.add_argument("--seed", type=int, default=42)
    # ── vLLM-specific args ──
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM for generation (faster inference). "
        "Uses truncated importance sampling to correct for policy mismatch.",
    )
    parser.add_argument(
        "--vllm_gpu_ratio",
        type=float,
        default=0.4,
        help="Fraction of GPU memory for vLLM (default 0.4). "
        "The rest is used by the training model.",
    )
    parser.add_argument(
        "--vllm_tp_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM. Use >1 to shard the model "
        "across multiple GPUs for faster generation (e.g. 2, 4, 8). "
        "No accelerate launch needed — just run with python directly.",
    )
    parser.add_argument(
        "--vllm_gpu_id",
        type=int,
        default=None,
        help="Pin vLLM to a specific GPU ID (e.g. 1 to put vLLM on GPU 1 "
        "while training runs on GPU 0). Only used when tp_size=1.",
    )
    parser.add_argument(
        "--is_clip",
        type=float,
        default=5.0,
        help="Truncated importance sampling clip ratio c. "
        "Weights are clipped to [1/c, c]. (paper default: 5)",
    )
    args = parser.parse_args()

    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        log_with="wandb" if args.wandb else None,
    )

    if args.max_seq_len <= args.max_gen_len:
        raise ValueError(
            f"max_seq_len ({args.max_seq_len}) must be larger than max_gen_len ({args.max_gen_len})"
        )

    set_seed(args.seed)

    if args.wandb:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
        )

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
    # Move model to GPU immediately to ensure SVD (in TinyLoRAModel) runs on GPU.
    base_model.to(accelerator.device)

    if not args.no_gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        # Gradient checkpointing requires at least one input with requires_grad=True.
        # Since TinyLoRA freezes all base weights, we need to explicitly enable this
        # on the input embeddings, otherwise backward() fails.
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

    # Enable grad for trainable params
    for p in model.trainable_parameters():
        p.requires_grad = True

    optimizer = AdamW(model.trainable_parameters(), lr=args.lr)

    # Keep a reference to the raw model for generation & saving.
    # DDP wraps the model, but raw_model shares the same weights.
    raw_model = model

    # ── Initialize vLLM engine ──
    vllm_engine = None
    if args.use_vllm:
        if accelerator.num_processes > 1:
            print(
                "[vLLM] WARNING: Running with multiple DDP processes. "
                "This is not recommended with --use_vllm. Use --vllm_tp_size "
                "instead to leverage multiple GPUs for generation: "
                "python train_grpo.py --use_vllm --vllm_tp_size N"
            )
        if accelerator.is_main_process:
            print("[vLLM] Initializing vLLM engine for inference...")
            print(f"[vLLM] GPU memory utilization: {args.vllm_gpu_ratio}")
            print(f"[vLLM] Tensor parallel size: {args.vllm_tp_size}")
            if args.vllm_gpu_id is not None:
                print(f"[vLLM] Pinned to GPU: {args.vllm_gpu_id}")
            vllm_engine = init_vllm_engine(
                model_name=args.model,
                gpu_memory_utilization=args.vllm_gpu_ratio,
                max_model_len=args.max_seq_len,
                enforce_eager=True,
                tp_size=args.vllm_tp_size,
                gpu_id=args.vllm_gpu_id,
            )
            print("[vLLM] Engine initialized successfully")

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
        pin_memory=True,
    )

    if accelerator.is_main_process:
        print(f"  Train: {len(dataset)} examples")

    # ── Prepare with Accelerator ──
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # ── Training loop ──
    step = 0
    # Calculate approximate total steps (len(dataloader) is per process)
    total_steps = len(dataloader) * args.epochs

    pbar = tqdm(
        total=total_steps, disable=not accelerator.is_main_process, desc="Training"
    )

    for epoch in range(args.epochs):
        for batch_questions, batch_answers in dataloader:

            # Form prompts
            prompts = [format_prompt(q, tokenizer) for q in batch_questions]

            # ── Generation phase ──
            importance_weights = None

            if args.use_vllm:
                # ── vLLM path: generate with base model, IS corrects mismatch ──
                # vLLM uses base model weights (no hot-swap needed).
                # The IS correction handles the gap between vLLM's policy
                # (base model) and the true TinyLoRA policy.

                if accelerator.is_main_process:
                    if step % args.log_every == 0:
                        print(
                            f"[vLLM][Step {step}] Generating {len(prompts)}×{args.k} completions..."
                        )
                    completions, gen_lengths = generate_completions_vllm(
                        vllm_engine,
                        prompts,
                        args.k,
                        args.max_gen_len,
                    )
                    if step % args.log_every == 0:
                        print(f"[vLLM][Step {step}] Generation complete")


                # 5. Compute truncated importance sampling weights
                #    We need log π_θ(y|x) / π_behavior(y|x) for each completion.
                #    Behavior policy = vLLM's base model (frozen weights, no LoRA delta).
                #    Target policy = true TinyLoRA model (base + delta).
                #    With 13 trainable params, the delta is tiny so IS weights ≈ 1.

                # Compute behavior log-probs (base model = vLLM's policy)
                # Temporarily zero out v to simulate the base model
                saved_vs = [v.data.clone() for v in raw_model.shared_vs]
                for v in raw_model.shared_vs:
                    v.data.zero_()
                behavior_log_probs = compute_behavior_log_probs(
                    raw_model,
                    tokenizer,
                    prompts,
                    completions,
                    args.max_seq_len,
                    accelerator.device,
                )
                # Restore v vectors
                for v, saved in zip(raw_model.shared_vs, saved_vs):
                    v.data.copy_(saved)

                # Compute target log-probs (true LoRA model)
                target_log_probs = compute_behavior_log_probs(
                    raw_model,
                    tokenizer,
                    prompts,
                    completions,
                    args.max_seq_len,
                    accelerator.device,
                )

                # IS weights with truncation: clip(exp(log_π_target - log_π_behavior), 1/c, c)
                importance_weights = []
                for t_lps, b_lps in zip(target_log_probs, behavior_log_probs):
                    t_t = torch.tensor(t_lps, dtype=torch.float32)
                    b_t = torch.tensor(b_lps, dtype=torch.float32)
                    log_ratio = t_t - b_t
                    # Truncated IS: clip to [1/c, c]
                    w = torch.exp(log_ratio).clamp(1.0 / args.is_clip, args.is_clip)
                    importance_weights.append(w.tolist())

                torch.cuda.empty_cache()
            else:
                # ── HuggingFace path (original behavior) ──
                # Use raw_model (shares weights with compiled/DDP model) for generation
                if not args.no_gradient_checkpointing:
                    # Disable GC for generation
                    if hasattr(raw_model.model, "gradient_checkpointing_disable"):
                        raw_model.model.gradient_checkpointing_disable()
                    raw_model.model.config.use_cache = True

                completions, _ = generate_completions_hf(
                    raw_model,
                    tokenizer,
                    prompts,
                    args.k,
                    args.max_gen_len,
                    args.max_seq_len,
                    device=accelerator.device,
                )

                # Re-enable gradient checkpointing
                if not args.no_gradient_checkpointing:
                    if hasattr(raw_model.model, "gradient_checkpointing_enable"):
                        raw_model.model.gradient_checkpointing_enable()
                    raw_model.model.config.use_cache = False

                # Free generation KV cache
                torch.cuda.empty_cache()

            # Compute rewards (CPU side)
            rewards = []
            for comps, answer in zip(completions, batch_answers):
                rews = [exact_match_reward(c, answer) for c in comps]
                rewards.append(rews)

            # GRPO update
            optimizer.zero_grad()

            # Pass accelerator to loss function for backward()
            avg_loss, n_samples = compute_grpo_loss_step(
                model,
                tokenizer,
                prompts,
                completions,
                rewards,
                accelerator=accelerator,
                max_seq_len=args.max_seq_len,
                micro_batch_size=args.micro_batch_size,
                importance_weights=importance_weights,
            )

            # Manual gradient sync across ranks.
            # We use no_sync inside compute_grpo_loss_step to avoid DDP deadlocks,
            # and all_reduce the tiny gradients here instead.
            if torch.distributed.is_initialized():
                for p in raw_model.trainable_parameters():
                    if p.grad is not None:
                        torch.distributed.all_reduce(
                            p.grad, op=torch.distributed.ReduceOp.AVG
                        )

            if n_samples > 0:
                num_microbatches = (
                    n_samples + args.micro_batch_size - 1
                ) // args.micro_batch_size
                if num_microbatches > 1:
                    for p in raw_model.trainable_parameters():
                        if p.grad is not None:
                            p.grad.div_(num_microbatches)

                optimizer.step()

            step += 1
            avg_reward = sum(r for rews in rewards for r in rews) / max(
                sum(len(r) for r in rewards), 1
            )

            if accelerator.is_main_process:
                pbar.update(1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", reward=f"{avg_reward:.3f}")
                if step % args.log_every == 0:
                    log_msg = f"[Step {step}] Loss: {avg_loss:.4f}  Avg reward: {avg_reward:.3f}"
                    if importance_weights is not None:
                        # Log average IS weight for monitoring
                        avg_iw = sum(w for ws in importance_weights for w in ws) / max(
                            sum(len(ws) for ws in importance_weights), 1
                        )
                        log_msg += f"  Avg IS weight: {avg_iw:.4f}"
                    tqdm.write(log_msg)

            if args.wandb:
                log_dict = {
                    "loss": avg_loss,
                    "reward": avg_reward,
                    "epoch": epoch,
                }
                if importance_weights is not None:
                    avg_iw = sum(w for ws in importance_weights for w in ws) / max(
                        sum(len(ws) for ws in importance_weights), 1
                    )
                    log_dict["is_weight_mean"] = avg_iw
                accelerator.log(log_dict, step=step)

    # ── Save ──
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        raw_model.save_adapter(args.output_dir)
        print("Done!")

    if args.wandb:
        accelerator.end_training()


if __name__ == "__main__":
    main()
