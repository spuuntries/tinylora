"""
Evaluation script for TinyLoRA-adapted models on math benchmarks.

Usage (HuggingFace inference):
    python eval.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --adapter ./tinylora_adapter \
        --dataset gsm8k \
        --max_gen_len 4096 \
        --batch_size 4 \
        --k 1

Usage (vLLM inference, much faster):
    python eval.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --adapter ./tinylora_adapter \
        --dataset gsm8k \
        --max_gen_len 4096 \
        --batch_size 4 \
        --k 1 \
        --use_vllm --vllm_gpu_ratio 0.9
"""

import argparse
import json
import re
import collections

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from tinylora import TinyLoRAModel


def extract_answer(text: str) -> str | None:
    """Extract the final numerical answer from model output."""
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


def check_correct(prediction: str, ground_truth: str) -> bool:
    pred = extract_answer(prediction)
    gt = extract_answer(ground_truth)
    if pred is None or gt is None:
        return False
    try:
        return abs(float(pred) - float(gt)) < 1e-5
    except ValueError:
        return pred.strip() == gt.strip()


@torch.no_grad()
def generate_completions_hf(
    model,
    tokenizer,
    prompts: list[str],
    k: int,
    max_gen_len: int,
) -> list[list[str]]:
    """
    Batched generation using HuggingFace generate() for k completions per prompt.
    """
    # 1. Set padding to LEFT for generation
    tokenizer.padding_side = "left"

    # 2. Tokenize all prompts
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
    ).to(next(model.parameters()).device)

    # 3. Generate
    # If k > 1, we MUST sample to get diversity.
    # If k = 1, we can use temperature=0 (greedy) for reproducibility/standard eval.
    do_sample = k > 1
    temperature = 0.7 if do_sample else 0.0

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_gen_len,
        temperature=temperature,
        do_sample=do_sample,
        top_p=0.95 if do_sample else 1.0,
        num_return_sequences=k,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    # 4. Decode and reshape
    all_texts = []

    # The outputs are ordered: [p1_k1, p1_k2, ..., p2_k1, p2_k2, ...]
    prompt_lens = inputs["input_ids"].shape[1]

    for i in range(len(prompts)):
        start_idx = i * k
        end_idx = start_idx + k

        group_texts = []
        for j in range(start_idx, end_idx):
            # Extract only the new tokens
            gen_ids = outputs[j, prompt_lens:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            group_texts.append(text)

        all_texts.append(group_texts)

    # Reset padding side
    tokenizer.padding_side = "right"

    return all_texts


def generate_completions_vllm(
    llm,
    prompts: list[str],
    k: int,
    max_gen_len: int,
) -> list[list[str]]:
    """
    Batched generation using vLLM for k completions per prompt.
    """
    from vllm import SamplingParams

    do_sample = k > 1
    temperature = 0.7 if do_sample else 0.0

    sampling_params = SamplingParams(
        n=k,
        temperature=temperature,
        top_p=0.95 if do_sample else 1.0,
        max_tokens=max_gen_len,
    )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    all_texts = []
    for output in outputs:
        group_texts = [comp.text for comp in output.outputs]
        all_texts.append(group_texts)

    return all_texts


def evaluate_gsm8k(
    model_or_llm,
    tokenizer,
    split="test",
    max_gen_len=4096,
    max_samples=None,
    batch_size=4,
    k=1,
    use_vllm=False,
):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    # Metrics
    total_instances = 0
    total_correct_greedy = 0  # For k=1 or just taking the first completion
    total_pass_at_k = 0  # At least one correct
    total_vote_correct = 0  # Majority vote

    # Batched iteration
    # Convert dataset to list for easier slicing
    data = list(ds)

    # Progress bar
    n_batches = (len(data) + batch_size - 1) // batch_size
    pbar = tqdm(total=n_batches, desc="Evaluating")

    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        questions = [row["question"] for row in batch]
        answers = [row["answer"] for row in batch]

        # Prepare prompts
        prompts = []
        for q in questions:
            messages = [
                {
                    "role": "system",
                    "content": "Solve the following math problem step by step. Put your final answer after ####.",
                },
                {"role": "user", "content": q},
            ]
            prompts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        # Generate
        if use_vllm:
            completions_batch = generate_completions_vllm(
                model_or_llm, prompts, k, max_gen_len
            )
        else:
            completions_batch = generate_completions_hf(
                model_or_llm, tokenizer, prompts, k, max_gen_len
            )

        # Evaluate
        for comps, gt_ans in zip(completions_batch, answers):
            # Check correctness for each completion
            results = [check_correct(c, gt_ans) for c in comps]

            # 1. Greedy / First (Standard Accuracy)
            if results[0]:
                total_correct_greedy += 1

            # 2. Pass@k (At least one correct)
            if any(results):
                total_pass_at_k += 1

            # 3. Majority Vote
            if k > 1:
                # Extract all answers
                extracted_answers = [extract_answer(c) for c in comps]
                # Filter None
                valid_answers = [a for a in extracted_answers if a is not None]
                if valid_answers:
                    counter = collections.Counter(valid_answers)
                    most_common, _ = counter.most_common(1)[0]
                    if check_correct(most_common, gt_ans):
                        total_vote_correct += 1

            total_instances += 1

        pbar.update(1)
        # Update desc with current accuracy
        curr_acc = total_correct_greedy / total_instances
        pbar.set_postfix(acc=f"{curr_acc:.1%}")

    pbar.close()

    accuracy = total_correct_greedy / total_instances if total_instances > 0 else 0.0
    pass_at_k = total_pass_at_k / total_instances if total_instances > 0 else 0.0
    vote_acc = total_vote_correct / total_instances if total_instances > 0 else 0.0

    print(f"\n=== Results (k={k}) ===")
    print(f"Dataset: GSM8K ({split})")
    print(f"Total: {total_instances}")
    print(
        f"Accuracy (Greedy/First): {total_correct_greedy}/{total_instances} = {accuracy:.1%}"
    )

    metrics = {
        "accuracy": accuracy,
        "correct": total_correct_greedy,
        "total": total_instances,
    }

    if k > 1:
        print(f"Pass@{k}: {pass_at_k:.1%}")
        print(f"Majority Vote Accuracy: {vote_acc:.1%}")
        metrics["pass_at_k"] = pass_at_k
        metrics["vote_accuracy"] = vote_acc

    return metrics


def main():
    parser = argparse.ArgumentParser(description="TinyLoRA Evaluation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to TinyLoRA adapter directory (optional)",
    )
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_gen_len", type=int, default=4096)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit evaluation to N samples (for quick testing)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of completions per prompt (1 for greedy, >1 for sampling)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Save results to JSON file"
    )
    # ── vLLM-specific args ──
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM for generation (much faster for eval)",
    )
    parser.add_argument(
        "--vllm_gpu_ratio",
        type=float,
        default=0.9,
        help="Fraction of GPU memory for vLLM (default 0.9, "
        "higher than training since no training model needed)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_vllm:
        # ── vLLM path: merge adapter into weights, then load into vLLM ──
        print("[vLLM] Using vLLM for evaluation")

        if args.adapter:
            # Load base model on CPU first, apply TinyLoRA, merge, then
            # save merged weights to a temp dir for vLLM to load.
            print("[vLLM] Merging adapter into model weights...")
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True,
            )
            with open(f"{args.adapter}/config.json") as f:
                config = json.load(f)

            tiny_model = TinyLoRAModel(
                base_model,
                rank=config["rank"],
                proj_dim=config["proj_dim"],
                n_tie=config["n_tie"],
                target_modules=config["target_modules"],
            )
            tiny_model.load_adapter(args.adapter)
            tiny_model.merge()

            # Save merged model to temp dir for vLLM
            import tempfile

            merged_dir = tempfile.mkdtemp(prefix="tinylora_merged_")
            print(f"[vLLM] Saving merged model to {merged_dir}...")
            base_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)

            # Free memory
            del base_model, tiny_model
            torch.cuda.empty_cache()

            model_path = merged_dir
        else:
            model_path = args.model

        import os

        os.environ["VLLM_USE_V1"] = "0"
        print("[vLLM] Forced V0 engine (VLLM_USE_V1=0)")
        from vllm import LLM

        llm = LLM(
            model=model_path,
            gpu_memory_utilization=args.vllm_gpu_ratio,
            max_model_len=args.max_gen_len + 1024,  # prompt + generation
            enforce_eager=True,
            dtype="bfloat16",
            tensor_parallel_size=1,
        )
        print("[vLLM] Engine initialized")

        results = evaluate_gsm8k(
            llm,
            tokenizer,
            split=args.split,
            max_gen_len=args.max_gen_len,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            k=args.k,
            use_vllm=True,
        )

        # Cleanup temp dir if created
        if args.adapter:
            import shutil

            shutil.rmtree(merged_dir, ignore_errors=True)
    else:
        # ── HuggingFace path (original behavior) ──
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )

        if args.adapter:
            # Load adapter config to reconstruct TinyLoRA
            with open(f"{args.adapter}/config.json") as f:
                config = json.load(f)
            model = TinyLoRAModel(
                base_model,
                rank=config["rank"],
                proj_dim=config["proj_dim"],
                n_tie=config["n_tie"],
                target_modules=config["target_modules"],
            )
            model.load_adapter(args.adapter)
            model.merge()  # Merge for fast inference
            print("[Eval] Merged adapter into model weights")
        else:
            model = base_model

        results = evaluate_gsm8k(
            model,
            tokenizer,
            split=args.split,
            max_gen_len=args.max_gen_len,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            k=args.k,
            use_vllm=False,
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
