#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterative STaR (Self-Taught Reasoner) Training Pipeline for CyberRott
====================================================================
Implements closed-loop reasoning self-evolution via RLVR and Rejection Sampling:
1. Inference Rollout (N=4 to 8 exploratory trajectories with <think> reasoning).
2. Deterministic Verification and Rejection Sampling (STaRVerifier).
3. Curated Self-Training Corpus (Strict filtering, quality ranking, and Golden Anchors).
4. Iterative SFT / LoRA Update (TRL SFTTrainer + PEFT).
5. Self-Evolution Loop: The upgraded checkpoint at iteration i generates rollouts for iteration i+1.

Supports high-performance inference rollout backends (vLLM, native Transformers) and
deterministic non-GPU simulation (--dry_run / mock) for end-to-end CPU verification.
"""

import os
import sys
import gc
import re
import json
import time
import inspect
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from scripts.star_verifier import STaRVerifier, VerificationResult, CHOICE_MAP
except ImportError:
    from star_verifier import STaRVerifier, VerificationResult, CHOICE_MAP

import torch

# CUDA memory allocation optimizations and W&B deactivation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_DISABLED"] = "true"


# Backbone Scaling Catalog (v1.4)
SUPPORTED_BACKBONES: Dict[str, Dict[str, Any]] = {
    "phi4-mini": {
        "model_id": "microsoft/Phi-4-mini-instruct",
        "target_modules": "all-linear",
        "stop_tokens": ["<|end|>", "<|endoftext|>"],
        "default_max_seq_length": 4096,
        "description": "Microsoft Phi-4-mini (3.82B) dense reasoning architecture",
    },
    "qwen2.5-coder-7b": {
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "default_max_seq_length": 4096,
        "description": "Qwen 2.5 Coder 7B Instruct specialized in code audit & reversing",
    },
    "phi4-14b": {
        "model_id": "microsoft/Phi-4",
        "target_modules": "all-linear",
        "stop_tokens": ["<|im_end|>", "<|end|>", "<|endoftext|>"],
        "default_max_seq_length": 4096,
        "description": "Microsoft Phi-4 14B deep deductive reasoning model",
    },
}


def resolve_backbone_config(backbone_or_model: str) -> Dict[str, Any]:
    """Resolve a chave de backbone ou ID de modelo Hugging Face para a configuração canônica."""
    if not backbone_or_model:
        return SUPPORTED_BACKBONES["phi4-mini"]
    key = backbone_or_model.lower().strip()
    if key in SUPPORTED_BACKBONES:
        return SUPPORTED_BACKBONES[key]
    for b_key, b_cfg in SUPPORTED_BACKBONES.items():
        if b_cfg["model_id"].lower() == key:
            return b_cfg
    return {
        "model_id": backbone_or_model,
        "target_modules": "all-linear",
        "stop_tokens": ["<|end|>", "<|endoftext|>", "<|im_end|>"],
        "default_max_seq_length": 4096,
        "description": f"Custom architecture: {backbone_or_model}",
    }


def build_lora_config(
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: Any = "all-linear",
    use_dora: bool = False,
    bias: str = "none",
    task_type: Any = None,
) -> Any:
    """Instancia LoraConfig com suporte transparente a DoRA e target modules flexíveis."""
    from peft import LoraConfig, TaskType
    t_type = task_type if task_type is not None else TaskType.CAUSAL_LM
    lora_kwargs = {
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": target_modules,
        "bias": bias,
        "task_type": t_type,
    }
    if use_dora:
        sig = inspect.signature(LoraConfig.__init__).parameters
        if "use_dora" in sig:
            lora_kwargs["use_dora"] = True
    return LoraConfig(**lora_kwargs)


def setup_hardware_optimizations(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """
    Detects and applies modern hardware accelerations for modern GPUs (A100, H100, L4):
    - TF32 across cuDNN and matmul (up to 3x tensor throughput without precision loss)
    - Preserves micro-batch at 2 and grad_accum at 8 (due to Phi-4's 200k vocabulary)
    - Automatic detection and preference for FlashAttention-2
    """
    profile: Dict[str, Any] = {
        "device_name": "CPU",
        "vram_gb": 0.0,
        "is_ampere_or_newer": False,
        "is_high_vram": False,
        "recommended_attn": "sdpa",
    }
    if not torch.cuda.is_available():
        return profile

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    major, _ = torch.cuda.get_device_capability(0)

    profile["device_name"] = device_name
    profile["vram_gb"] = round(total_vram_gb, 1)
    profile["is_ampere_or_newer"] = (major >= 8)
    profile["is_high_vram"] = (total_vram_gb >= 35.0)  # A100 40GB/80GB, H100 80GB

    try:
        import flash_attn  # noqa: F401
        profile["recommended_attn"] = "flash_attention_2"
    except ImportError:
        profile["recommended_attn"] = "sdpa"

    if args is not None and getattr(args, "auto_optimize", True):
        if profile["is_high_vram"]:
            if args.attn_implementation == "sdpa" and profile["recommended_attn"] == "flash_attention_2":
                args.attn_implementation = "flash_attention_2"
                print("[A100 Auto-Optimization] Automatically activated 'flash_attention_2'.")

    return profile


# ==============================================================================
# Command Line Interface (CLI) Arguments
# ==============================================================================

def parse_args(args_list: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="CyberRott - Iterative STaR / Rejection Sampling Training Pipeline"
    )

    # 1. Problems and Datasets
    data_group = parser.add_argument_group("Problems and Datasets")
    data_group.add_argument(
        "--problem_file", "--problem-file",
        dest="problem_file",
        type=str,
        default="data/dataset_cyber_ready.jsonl",
        help="JSONL file containing problem suites/prompts for rollouts."
    )
    data_group.add_argument(
        "--wmdp_cyber", "--wmdp-cyber",
        dest="wmdp_cyber",
        action="store_true",
        default=False,
        help="Ingest WMDP-Cyber benchmark multiple-choice challenges for STaR self-distillation."
    )
    data_group.add_argument(
        "--wmdp_file", "--wmdp-file",
        dest="wmdp_file",
        type=str,
        default=None,
        help="Optional local path or HF identifier for WMDP-Cyber benchmark dataset."
    )
    data_group.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help="Limit the number of evaluated problems (ideal for rapid tests)."
    )
    data_group.add_argument(
        "--anchor_file", "--anchor-file",
        dest="anchor_file",
        type=str,
        default="data/dataset_cyber_ready.jsonl",
        help="JSONL file with Golden Anchor samples for knowledge retention."
    )
    data_group.add_argument(
        "--anchor_upsample", "--anchor-upsample",
        dest="anchor_upsample",
        type=int,
        default=5,
        help="Repetition/upsampling factor for anchor samples per iteration (0 to disable)."
    )

    # 2. STaR Cycle and Rejection Sampling Settings
    star_group = parser.add_argument_group("STaR Cycle and Rejection Sampling")
    star_group.add_argument(
        "--star_iterations", "--iterations", "--star-iterations",
        dest="star_iterations",
        type=int,
        default=1,
        help="Number of iterations for the STaR loop (Rollout -> Verifier -> SFT Update)."
    )
    star_group.add_argument(
        "--num_candidates", "--num-candidates",
        dest="num_candidates",
        type=int,
        default=4,
        help="Number N of exploratory trajectories generated per problem (default: 4, v1.4: 16)."
    )
    star_group.add_argument(
        "--max_verified_per_problem",
        type=int,
        default=1,
        help="Maximum verified trajectories retained per problem (Top-K approved, default: 1)."
    )
    star_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for logical exploration during rollout generation."
    )
    star_group.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling for rollout generation."
    )
    star_group.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum new tokens generated per exploratory trajectory."
    )
    star_group.add_argument(
        "--rollout_backend", "--backend", "--rollout-backend",
        dest="rollout_backend",
        type=str,
        default="hf",
        choices=["hf", "vllm", "mock"],
        help="Inference backend for rollouts ('vllm' for high-throughput batching, 'hf' for native, 'mock' for simulation)."
    )
    star_group.add_argument(
        "--dry_run", "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Deterministic rapid simulation mode (runs pipeline on CPU without loading 4B weights)."
    )

    # 3. Deterministic Verifier Settings (STaRVerifier)
    verifier_group = parser.add_argument_group("Deterministic Verifier")
    verifier_group.add_argument(
        "--min_think_chars",
        type=int,
        default=80,
        help="Minimum characters required inside the <think>...</think> deliberation block."
    )
    verifier_group.add_argument(
        "--strict_code_syntax",
        action="store_true",
        default=True,
        help="Strictly reject trajectories containing code blocks with syntax errors."
    )
    verifier_group.add_argument(
        "--strict_ground_truth",
        action="store_true",
        default=True,
        help="Strictly reject trajectories failing multiple-choice key or expected CWE taxonomy."
    )
    verifier_group.add_argument(
        "--include_prm",
        action="store_true",
        default=False,
        help="Include Step-Level Process Reward Model (PRM) in trajectory scoring and rejection."
    )
    verifier_group.add_argument(
        "--w_prm",
        type=float,
        default=0.25,
        help="Weight assigned to PRM score during composite ranking (default: 0.25)."
    )

    # 4. Model and LoRA Settings
    model_group = parser.add_argument_group("Base Model and LoRA")
    model_group.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["phi4-mini", "qwen2.5-coder-7b", "phi4-14b"],
        help="Pre-configured backbone architecture from SUPPORTED_BACKBONES."
    )
    model_group.add_argument(
        "--model_name_or_path", "--model_path", "--model-path", "--model-name-or-path",
        dest="model_name_or_path",
        type=str,
        default="microsoft/Phi-4-mini-instruct",
        help="Initial base model or prior LoRA checkpoint for the first STaR iteration."
    )
    model_group.add_argument(
        "--initial_adapter", "--adapter_path", "--adapter-path",
        dest="initial_adapter",
        type=str,
        default=None,
        help="Optional path to existing LoRA adapter (e.g. CyberRott 1.0 checkpoint)."
    )
    model_group.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model numerical precision (default: bfloat16)."
    )
    model_group.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention kernel implementation."
    )
    model_group.add_argument(
        "--lora_r",
        type=int,
        default=32,
        help="LoRA rank (default: 32)."
    )
    model_group.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha scaling factor (default: 64)."
    )
    model_group.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate (default: 0.05)."
    )
    model_group.add_argument(
        "--target_modules",
        type=str,
        default="all-linear",
        help="Target modules for LoRA (default: 'all-linear')."
    )
    model_group.add_argument(
        "--use_dora",
        action="store_true",
        default=False,
        help="Enable Weight-Decomposed Low-Rank Adaptation (DoRA) for enhanced capacity."
    )
    model_group.add_argument(
        "--v1_4", "--v1.4", "--preset_v1_4",
        dest="v1_4",
        action="store_true",
        default=False,
        help="Activate CyberRott v1.4 self-distillation defaults (DoRA, LoRA r=64/alpha=128, N=16 candidates, 2048 tokens)."
    )

    # 5. SFT Training Hyperparameters
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument(
        "--output_dir", "--output-dir",
        dest="output_dir",
        type=str,
        default="./CyberRott-Phi4-STaR",
        help="Output directory for checkpoints, curated datasets, and telemetry."
    )
    train_group.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of SFT fine-tuning epochs per STaR iteration."
    )
    train_group.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Per-device batch size for SFT training."
    )
    train_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps."
    )
    train_group.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate for fine-tuning."
    )
    train_group.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence context length for SFTTrainer."
    )
    train_group.add_argument(
        "--logging_steps",
        type=int,
        default=5,
        help="Loss logging frequency in steps."
    )
    train_group.add_argument(
        "--optim",
        type=str,
        default="adamw_torch_fused",
        help="PyTorch optimizer choice."
    )
    train_group.add_argument(
        "--auto_optimize",
        action="store_true",
        default=True,
        help="Auto-optimize batch size, TF32, and FlashAttention for A100/H100/L4 (default: True)."
    )

    parsed = parser.parse_args(args_list)
    if getattr(parsed, "v1_4", False):
        parsed.use_dora = True
        parsed.include_prm = True
        if parsed.num_candidates == 4:
            parsed.num_candidates = 16
        if parsed.lora_r == 32:
            parsed.lora_r = 64
        if parsed.lora_alpha == 64:
            parsed.lora_alpha = 128
        if parsed.max_new_tokens == 1024:
            parsed.max_new_tokens = 2048
    if getattr(parsed, "backbone", None):
        cfg = resolve_backbone_config(parsed.backbone)
        if parsed.model_name_or_path == "microsoft/Phi-4-mini-instruct":
            parsed.model_name_or_path = cfg["model_id"]
        if parsed.target_modules == "all-linear":
            tm = cfg["target_modules"]
            parsed.target_modules = "all-linear" if tm == "all-linear" else ",".join(tm)
    return parsed


# ==============================================================================
# Problem Suite Ingestion and Prompt Formatting
# ==============================================================================

DEFAULT_SYSTEM_PROMPT = (
    "You are CyberRott, a specialized reasoning AI for offensive, defensive, and "
    "vulnerability analysis with deep analytical reasoning."
)


def load_problem_suite(problem_file: str, max_problems: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Loads cybersecurity problem suite from a JSONL file, supporting ChatML
    ('messages') and raw problem schema ('prompt', 'question', 'expected_cwe', 'answer', etc.).
    """
    if not os.path.exists(problem_file):
        fallback = os.path.join("data", problem_file)
        if os.path.exists(fallback):
            print(f"[*] Problem suite fallback: '{problem_file}' -> '{fallback}'")
            problem_file = fallback
        else:
            raise FileNotFoundError(f"Problem file '{problem_file}' not found.")

    problems = []
    with open(problem_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue

            problem_item: Dict[str, Any] = {
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "user_prompt": "",
                "reference_answer": None,
                "metadata": dict(data),
            }

            # Case 1: ChatML format with 'messages' list
            if "messages" in data and isinstance(data["messages"], list):
                for msg in data["messages"]:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if role == "system":
                        problem_item["system_prompt"] = content
                    elif role == "user":
                        problem_item["user_prompt"] = content
                    elif role == "assistant":
                        problem_item["reference_answer"] = content

            # Case 2: Multiple-choice benchmark format (WMDP / SecQA)
            elif "question" in data and "choices" in data:
                q = data["question"].strip()
                choices = data["choices"]
                choices_formatted = "\n".join([f"{CHOICE_MAP.get(i, str(i))}. {c}" for i, c in enumerate(choices)])
                user_content = (
                    f"{q}\n\n{choices_formatted}\n\n"
                    "Instructions:\n"
                    "1. Formulate your technical analysis and reasoning enclosed within <think>...</think> tags.\n"
                    "2. After closing </think>, state the correct option letter in the exact format: 'Final Answer: [OPTION]'."
                )
                problem_item["user_prompt"] = user_content
                if "answer" in data:
                    problem_item["metadata"]["expected_answer"] = CHOICE_MAP.get(data["answer"], str(data["answer"]))

            # Case 3: Direct 'prompt' format
            elif "prompt" in data:
                problem_item["user_prompt"] = data["prompt"]
                if "ground_truth" in data:
                    problem_item["reference_answer"] = data["ground_truth"]

            if problem_item["user_prompt"]:
                problems.append(problem_item)
                if max_problems and len(problems) >= max_problems:
                    break

    return problems


def load_wmdp_cyber_prompts(
    dataset_name_or_file: str = "cais/wmdp",
    split: str = "test",
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Ingestão de desafios de domínio e benchmark WMDP-Cyber (cais/wmdp / wmdp-cyber).
    Formata questões de múltipla escolha para o loop de auto-destilação STaR.
    """
    problems = []
    # 1. Tenta carregar se for arquivo JSONL local
    target_path = dataset_name_or_file if os.path.exists(dataset_name_or_file) else os.path.join("data", dataset_name_or_file)
    if os.path.exists(target_path):
        items = load_problem_suite(target_path, max_problems=max_samples)
        for it in items:
            it.setdefault("metadata", {})["benchmark"] = "WMDP-Cyber"
        return items

    # 2. Ingestão via datasets (Hugging Face)
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name_or_file, "wmdp-cyber", split=split)
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            q = item.get("question", "").strip()
            choices = item.get("choices", [])
            answer = item.get("answer")
            choices_formatted = "\n".join([f"{CHOICE_MAP.get(idx, str(idx))}. {c}" for idx, c in enumerate(choices)])
            user_content = (
                f"{q}\n\n{choices_formatted}\n\n"
                "Instructions:\n"
                "1. Formulate your technical analysis and reasoning enclosed within <think>...</think> tags.\n"
                "2. After closing </think>, state the correct option letter in the exact format: 'Final Answer: [OPTION]'."
            )
            expected_letter = CHOICE_MAP.get(answer, str(answer).strip().upper()) if answer is not None else None
            problems.append({
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "user_prompt": user_content,
                "reference_answer": None,
                "metadata": {
                    "question": q,
                    "choices": choices,
                    "answer": answer,
                    "expected_answer": expected_letter,
                    "benchmark": "WMDP-Cyber",
                }
            })
    except Exception as e:
        print(f"[Warning] Unable to load WMDP-Cyber from '{dataset_name_or_file}': {e}")
    return problems


def load_gold_anchors(anchor_file: str, upsample_factor: int = 1) -> List[Dict[str, Any]]:
    """
    Loads Golden Anchor samples in ChatML format and applies upsampling,
    ensuring knowledge retention and preventing catastrophic forgetting.
    """
    if not anchor_file or upsample_factor <= 0:
        return []

    if not os.path.exists(anchor_file):
        fallback = os.path.join("data", anchor_file)
        if os.path.exists(fallback):
            print(f"[*] STaR anchor fallback: '{anchor_file}' -> '{fallback}'")
            anchor_file = fallback
        else:
            return []

    anchors = []
    with open(anchor_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "messages" in data:
                    anchors.append(data)
            except Exception:
                continue

    if not anchors:
        return []

    upsampled = []
    for _ in range(upsample_factor):
        upsampled.extend(anchors)

    return upsampled


# ==============================================================================
# Phase 1: Exploratory Rollout Generator (Inference Rollout)
# ==============================================================================

class RolloutGenerator:
    """
    Manages generation of N candidate trajectories per problem.
    Supports backends: 'vllm', 'hf' (Hugging Face Transformers + PEFT), and 'mock'.
    Optimized for OOM prevention with dynamic VRAM deallocation.
    """

    def __init__(
        self,
        backend: str,
        model_name_or_path: str,
        adapter_path: Optional[str] = None,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 1024,
    ):
        self.backend = backend
        self.model_name_or_path = model_name_or_path
        self.adapter_path = adapter_path
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens

        self.tokenizer = None
        self.model = None
        self.vllm_engine = None

        self._hf_model = None
        self._hf_tokenizer = None
        self._vllm_engine = None

        if self.backend != "mock":
            self._init_backend()

    def _init_hf(self):
        """
        Initializes Hugging Face backend with native LoRA Checkpoint support:
        If model_name_or_path contains adapter_config.json, loads the base model
        in bfloat16, attaches PeftModel adapter, and merges weights for fast inference.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        dtype = getattr(torch, self.torch_dtype)
        adapter_cfg_file = os.path.join(self.model_name_or_path, "adapter_config.json")
        is_lora_checkpoint = os.path.isdir(self.model_name_or_path) and os.path.isfile(adapter_cfg_file)

        if is_lora_checkpoint:
            try:
                with open(adapter_cfg_file, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                base_model_name = cfg_data.get("base_model_name_or_path", "microsoft/Phi-4-mini-instruct")
            except Exception:
                base_model_name = "microsoft/Phi-4-mini-instruct"

            print(f"[HF] LoRA checkpoint detected at: {self.model_name_or_path}")
            print(f"[HF] Loading base model '{base_model_name}' in {self.torch_dtype}...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=dtype,
                attn_implementation=self.attn_implementation,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                trust_remote_code=False,
            )
            print(f"[HF] Merging LoRA weights from '{self.model_name_or_path}' via merge_and_unload()...")
            peft_m = PeftModel.from_pretrained(base_model, self.model_name_or_path)
            self.model = peft_m.merge_and_unload()
            tokenizer_source = (
                self.model_name_or_path
                if os.path.exists(os.path.join(self.model_name_or_path, "tokenizer_config.json"))
                else base_model_name
            )
        else:
            print(f"[HF] Loading base model in {self.torch_dtype} from: {self.model_name_or_path}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=dtype,
                attn_implementation=self.attn_implementation,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                trust_remote_code=False,
            )
            tokenizer_source = self.model_name_or_path

            if self.adapter_path and os.path.exists(self.adapter_path):
                print(f"[HF] Loading and merging prior LoRA adapter from: {self.adapter_path}")
                peft_m = PeftModel.from_pretrained(self.model, self.adapter_path)
                self.model = peft_m.merge_and_unload()

        print(f"[HF] Loading tokenizer from: {tokenizer_source}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=False
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._hf_model = self.model
        self._hf_tokenizer = self.tokenizer
        self.model.eval()

    def _init_backend(self):
        """Initializes selected inference backend (vLLM or HF Transformers)."""
        if self.backend == "vllm":
            try:
                from vllm import LLM
                from transformers import AutoTokenizer

                print(f"[vLLM] Loading tokenizer from {self.model_name_or_path}...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=False
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self._hf_tokenizer = self.tokenizer

                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if torch.cuda.is_available() else 16
                gpu_util = 0.90 if total_mem >= 35.0 else 0.85
                print(f"[vLLM] Initializing vLLM engine in {self.torch_dtype} (VRAM: {total_mem:.1f}GB, gpu_memory_utilization={gpu_util})...")
                self.vllm_engine = LLM(
                    model=self.model_name_or_path,
                    trust_remote_code=False,
                    dtype=self.torch_dtype,
                    max_model_len=4096,
                    gpu_memory_utilization=gpu_util,
                    enforce_eager=False,
                )
                self._vllm_engine = self.vllm_engine
            except Exception as e:
                print(f"[Warning] Failed to initialize vLLM ({e}). Falling back to 'hf' backend...")
                self.backend = "hf"
                self._init_hf()

        elif self.backend == "hf":
            self._init_hf()

    def unload(self):
        """
        OOM prevention on GPU:
        Clears references to HF models, tokenizers, and vLLM engines,
        triggering garbage collection and emptying CUDA memory caches.
        """
        print("[RolloutGenerator] Unloading inference engine and releasing VRAM...")
        self.model = None
        self._hf_model = None
        self.tokenizer = None
        self._hf_tokenizer = None
        self.vllm_engine = None
        self._vllm_engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self):
        """Alias for unload()."""
        self.unload()

    def generate_candidates(
        self,
        problems: List[Dict[str, Any]],
        num_candidates: int
    ) -> List[List[str]]:
        """Generates N exploratory candidates per problem (alias for generate_rollouts)."""
        return self.generate_rollouts(problems, num_candidates)

    def generate_rollouts(
        self,
        problems: List[Dict[str, Any]],
        num_candidates: int
    ) -> List[List[str]]:
        """
        Generates N exploratory candidates for each problem in the list.
        Returns a nested list: rollouts_by_problem[i] = [cand_1, cand_2, ..., cand_N]
        """
        if self.backend == "mock":
            return self._mock_rollout_generation(problems, num_candidates)

        if self.backend == "vllm":
            return self._vllm_rollout_generation(problems, num_candidates)

        return self._hf_rollout_generation(problems, num_candidates)

    def _mock_rollout_generation(
        self,
        problems: List[Dict[str, Any]],
        num_candidates: int
    ) -> List[List[str]]:
        """
        Deterministic mock generation for full CPU verification (--dry_run).
        Synthesizes candidates containing <think> reasoning, answers, and code blocks.
        """
        all_rollouts = []

        for p in problems:
            cands = []
            user_prompt = p["user_prompt"]
            meta = p.get("metadata", {})

            expected_cwe = meta.get("expected_cwe") or ("CWE-89" if "SELECT" in user_prompt.upper() else "CWE-121")
            if isinstance(expected_cwe, list):
                expected_cwe = expected_cwe[0]

            expected_ans = meta.get("expected_answer") or meta.get("answer") or "A"
            if isinstance(expected_ans, int):
                expected_ans = CHOICE_MAP.get(expected_ans, "A")

            # Candidate 1: Perfect trajectory (Passes all 3 verifiers)
            cand_perfect = (
                f"<think>\n"
                f"**Surface Analysis**: Evaluating target code and security boundary requirements.\n"
                f"**Entry Point / Attack Vector Identification**: Untrusted input flows directly into program logic.\n"
                f"**Data Flow / Internal Mechanics (Taint Analysis)**: Taint tracking indicates absence of bounds checks.\n"
                f"**Vulnerable Sink / Control Flaw Detection**: Dangerous sink identified and mapped to {expected_cwe}.\n"
                f"**Sanitization and Mitigation Assessment**: No sanitization exists between source and sink.\n"
                f"**Vulnerability Verdict / Anomaly Classification**: The code is vulnerable to {expected_cwe}.\n"
                f"**Potential Impact**: Risk of unauthorized memory corruption or sensitive data exfiltration.\n"
                f"</think>\n\n"
                f"FACT: Technical analysis confirms vulnerability classified as {expected_cwe}.\n"
                f"Resposta Final: [{expected_ans}]\n"
                f"Final Answer: [{expected_ans}]\n\n"
                f"REMEDIATION / DEFENSIVE IMPLEMENTATION:\n"
                f"```python\n"
                f"def sanitize_and_execute(safe_param):\n"
                f"    # Parameterized verification routine\n"
                f"    return True\n"
                f"```"
            )
            cands.append(cand_perfect)

            # Candidate 2: Wrong Answer / Divergent CWE (Must fail Ground Truth)
            other_ans = "B" if expected_ans != "B" else "C"
            cand_bad_gt = (
                f"<think>\n"
                f"Surface analysis and entry points inspected superficially.\n"
                f"Taint propagation analysis inconclusive.\n"
                f"</think>\n\n"
                f"FACT: Hypothetical vulnerability CWE-999.\n"
                f"Resposta Final: [{other_ans}]\n"
                f"Final Answer: [{other_ans}]"
            )
            cands.append(cand_bad_gt)

            # Candidate 3: Python Syntax Error (Must fail Code Syntax Sandbox)
            cand_bad_syntax = (
                f"<think>\n"
                f"Full surface analysis locating sink for {expected_cwe}.\n"
                f"Taint flow and mitigation posture evaluated.\n"
                f"Potential impact assessed.\n"
                f"</think>\n\n"
                f"FACT: Identified {expected_cwe}.\n"
                f"Resposta Final: [{expected_ans}]\n"
                f"Final Answer: [{expected_ans}]\n\n"
                f"```python\n"
                f"def broken_syntax(:\n"
                f"    pass\n"
                f"```"
            )
            cands.append(cand_bad_syntax)

            # Additional candidates if N > 3
            for i in range(3, num_candidates):
                cand_extra = (
                    f"<think>\n"
                    f"**Surface Analysis**: Analytical variant {i}.\n"
                    f"**Entry Point Identification**: Input analyzed with boundary validation.\n"
                    f"**Data Flow Analysis**: Validated taint propagation down to critical sink.\n"
                    f"**Vulnerability Verdict**: Conforms to {expected_cwe}.\n"
                    f"**Potential Impact**: Controlled exposure mitigated through immediate patch.\n"
                    f"</think>\n\n"
                    f"FACT: Verified diagnosis corresponding to {expected_cwe}.\n"
                    f"Resposta Final: [{expected_ans}]\n"
                    f"Final Answer: [{expected_ans}]\n\n"
                    f"REMEDIATION:\n"
                    f"```python\n"
                    f"def patch_variant_{i}(val):\n"
                    f"    return str(val).strip()\n"
                    f"```"
                )
                cands.append(cand_extra)

            all_rollouts.append(cands[:num_candidates])

        return all_rollouts

    def _vllm_rollout_generation(
        self,
        problems: List[Dict[str, Any]],
        num_candidates: int
    ) -> List[List[str]]:
        """High-throughput batched generation via vLLM."""
        from vllm import SamplingParams

        prompts = []
        for p in problems:
            messages = [
                {"role": "system", "content": p["system_prompt"]},
                {"role": "user", "content": p["user_prompt"]}
            ]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(formatted)

        sampling_params = SamplingParams(
            n=num_candidates,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
            stop=["<|im_end|>", "<|endoftext|>"]
        )

        outputs = self.vllm_engine.generate(prompts, sampling_params)
        all_rollouts = []
        for out in outputs:
            problem_cands = [sub_out.text.strip() for sub_out in out.outputs]
            all_rollouts.append(problem_cands)

        return all_rollouts

    def _hf_rollout_generation(
        self,
        problems: List[Dict[str, Any]],
        num_candidates: int
    ) -> List[List[str]]:
        """Native generation via Hugging Face Transformers."""
        all_rollouts = []

        for p in problems:
            messages = [
                {"role": "system", "content": p["system_prompt"]},
                {"role": "user", "content": p["user_prompt"]}
            ]
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(prompt_text, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                gen_outputs = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=num_candidates,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            cands = []
            for out_seq in gen_outputs:
                gen_text = self.tokenizer.decode(out_seq[input_len:], skip_special_tokens=True).strip()
                cands.append(gen_text)

            all_rollouts.append(cands)

        return all_rollouts


# ==============================================================================
# Phase 2: Deterministic Verification and Rejection Sampling (STaR)
# ==============================================================================

def execute_rejection_sampling(
    problems: List[Dict[str, Any]],
    rollouts_by_problem: List[List[str]],
    verifier: STaRVerifier,
    max_verified_per_problem: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes deterministic verification across all generated trajectories,
    prunes invalid solutions, and selects top-ranked approved trajectories
    to build the ChatML self-training dataset.
    """
    verified_corpus: List[Dict[str, Any]] = []

    telemetry: Dict[str, Any] = {
        "total_problems": len(problems),
        "total_rollouts": 0,
        "total_verified_passed": 0,
        "accepted_problems_count": 0,
        "rejection_breakdown": {
            "format_failed": 0,
            "ground_truth_failed": 0,
            "code_syntax_failed": 0,
            "refusals_detected": 0,
        },
        "scores_passed": [],
        "scores_all": [],
    }

    for prob, candidates in zip(problems, rollouts_by_problem):
        meta = prob.get("metadata", {})
        ranked_candidates = []

        for cand in candidates:
            telemetry["total_rollouts"] += 1
            res = verifier.verify_candidate(meta, cand)
            telemetry["scores_all"].append(res.score)

            if not res.format_ok:
                telemetry["rejection_breakdown"]["format_failed"] += 1
            if not res.ground_truth_ok:
                telemetry["rejection_breakdown"]["ground_truth_failed"] += 1
            if not res.code_syntax_ok:
                telemetry["rejection_breakdown"]["code_syntax_failed"] += 1
            if res.metrics.get("format", {}).get("refusals_detected"):
                telemetry["rejection_breakdown"]["refusals_detected"] += 1

            if res.passed:
                telemetry["total_verified_passed"] += 1
                telemetry["scores_passed"].append(res.score)
                ranked_candidates.append((cand, res))

        ranked_candidates.sort(key=lambda x: x[1].score, reverse=True)

        if ranked_candidates:
            telemetry["accepted_problems_count"] += 1
            top_k = ranked_candidates[:max_verified_per_problem]
            for cand, res in top_k:
                chatml_entry = {
                    "messages": [
                        {"role": "system", "content": prob["system_prompt"]},
                        {"role": "user", "content": prob["user_prompt"]},
                        {"role": "assistant", "content": cand}
                    ],
                    "star_verification": res.to_dict()
                }
                verified_corpus.append(chatml_entry)

    tot_roll = max(1, telemetry["total_rollouts"])
    telemetry["acceptance_rate"] = round(telemetry["total_verified_passed"] / tot_roll * 100, 2)
    telemetry["problem_solve_rate"] = round(telemetry["accepted_problems_count"] / max(1, len(problems)) * 100, 2)
    telemetry["mean_score_passed"] = (
        round(sum(telemetry["scores_passed"]) / len(telemetry["scores_passed"]), 4)
        if telemetry["scores_passed"] else 0.0
    )
    telemetry["mean_score_all"] = (
        round(sum(telemetry["scores_all"]) / len(telemetry["scores_all"]), 4)
        if telemetry["scores_all"] else 0.0
    )

    return verified_corpus, telemetry


# ==============================================================================
# Phase 3: Iterative LoRA / SFT Fine-Tuning (run_sft_update)
# ==============================================================================

def run_sft_update(
    iteration: int,
    train_file: str,
    output_checkpoint_dir: str,
    args: argparse.Namespace,
    model_path: str,
    adapter_path_prev: Optional[str] = None,
) -> str:
    """
    Executes SFT LoRA fine-tuning on the STaR verified dataset.
    In --dry_run mode, simulates checkpoint generation.
    Supports continuous accumulation of learning on prior LoRA checkpoints.
    """
    print(f"\n[STaR Iteration {iteration}] Launching LoRA Fine-Tuning...")
    print(f"Curated Dataset:    {train_file}")
    print(f"Model / Checkpoint: {model_path}")
    print(f"Target Checkpoint:  {output_checkpoint_dir}")

    if args.dry_run:
        print("[Dry Run] Rapid SFT LoRA simulation active.")
        os.makedirs(output_checkpoint_dir, exist_ok=True)
        dummy_meta = {
            "iteration": iteration,
            "base_model_name_or_path": model_path,
            "train_file": train_file,
            "timestamp": datetime.now().isoformat(),
            "status": "dry_run_completed",
            "hyperparameters": {
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
            }
        }
        with open(os.path.join(output_checkpoint_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump(dummy_meta, f, indent=2)
        print(f"[Dry Run] Simulated checkpoint successfully saved to: {output_checkpoint_dir}")
        return output_checkpoint_dir

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, PeftModel
    from trl import SFTTrainer, SFTConfig

    dtype = getattr(torch, args.torch_dtype)

    adapter_cfg_file = os.path.join(model_path, "adapter_config.json")
    is_lora_checkpoint = os.path.isdir(model_path) and os.path.isfile(adapter_cfg_file)

    if is_lora_checkpoint:
        try:
            with open(adapter_cfg_file, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
            real_base = cfg_data.get("base_model_name_or_path", "microsoft/Phi-4-mini-instruct")
        except Exception:
            real_base = "microsoft/Phi-4-mini-instruct"

        print(f"[STaR LoRA Update] Prior LoRA checkpoint detected at '{model_path}'.")
        print(f"[STaR LoRA Update] Loading base model '{real_base}' in {args.torch_dtype}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            real_base,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=False,
        )
        print(f"[STaR LoRA Update] Merging prior adapter from '{model_path}' via merge_and_unload()...")
        model = PeftModel.from_pretrained(base_model, model_path).merge_and_unload()
        tokenizer_source = (
            model_path
            if os.path.exists(os.path.join(model_path, "tokenizer_config.json"))
            else real_base
        )
    else:
        print(f"[STaR SFT Update] Loading base model from '{model_path}' in {args.torch_dtype}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=False,
        )
        if adapter_path_prev and os.path.exists(adapter_path_prev) and adapter_path_prev != model_path:
            print(f"[STaR SFT Update] Merging additional prior adapter: {adapter_path_prev}")
            model = PeftModel.from_pretrained(model, adapter_path_prev).merge_and_unload()
        tokenizer_source = model_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_modules = "all-linear" if args.target_modules.strip().lower() == "all-linear" else [
        m.strip() for m in args.target_modules.split(",") if m.strip()
    ]
    peft_config = build_lora_config(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        use_dora=getattr(args, "use_dora", False),
    )

    dataset = load_dataset("json", data_files=train_file, split="train")

    def format_chat(batch):
        if "messages" in batch:
            return {
                "text": [
                    tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
                    for msg in batch["messages"]
                ]
            }
        return batch

    if "messages" in dataset.column_names:
        dataset = dataset.map(format_chat, batched=True)

    sft_config_params = inspect.signature(SFTConfig.__init__).parameters
    sft_kwargs = {
        "output_dir": output_checkpoint_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": "none",
        "learning_rate": args.learning_rate,
        "warmup_ratio": 0.05,
        "logging_steps": args.logging_steps,
        "save_strategy": "no",
        "bf16": (args.torch_dtype == "bfloat16" and torch.cuda.is_bf16_supported()),
        "fp16": (args.torch_dtype == "float16"),
        "optim": args.optim if torch.cuda.is_available() else "adamw_torch",
        "dataloader_num_workers": 2 if torch.cuda.is_available() else 0,
        "dataloader_pin_memory": True if torch.cuda.is_available() else False,
        "group_by_length": True,
    }

    if "max_length" in sft_config_params:
        sft_kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in sft_config_params:
        sft_kwargs["max_seq_length"] = args.max_seq_length

    if "dataset_text_field" in sft_config_params:
        sft_kwargs["dataset_text_field"] = "text"

    sft_config = SFTConfig(**sft_kwargs)

    sft_trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": peft_config,
        "args": sft_config,
    }
    if "processing_class" in sft_trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sft_trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()

    trainer.save_model(output_checkpoint_dir)
    tokenizer.save_pretrained(output_checkpoint_dir)

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_checkpoint_dir


execute_sft_iteration = run_sft_update


# ==============================================================================
# Main STaR Loop Orchestrator (run_star_pipeline)
# ==============================================================================

def run_star_pipeline(args: argparse.Namespace) -> Optional[str]:
    """
    Executes the full iterative STaR training cycle:
    1. Exploratory rollout with generation of N trajectories.
    2. Immediate VRAM cleanup (generator.unload() + empty_cache) to prevent OOM.
    3. Deterministic verification across 3 pillars and Rejection Sampling.
    4. Dataset curation and fusion with Golden Anchors.
    5. Incremental LoRA fine-tuning (run_sft_update) accumulating learning iteration-by-iteration.
    """
    hw_profile = setup_hardware_optimizations(args)
    start_total_time = time.time()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 75)
    print("🌟 CyberRott - Self-Taught Reasoner via Rejection Sampling (STaR)")
    print("=" * 75)
    print(f"GPU Hardware:             {hw_profile['device_name']} ({hw_profile['vram_gb']} GB VRAM)")
    print(f"Attention Kernel:         {args.attn_implementation} (TF32 Matmul: Active)")
    print(f"Effective Batch Size:     {args.per_device_train_batch_size} x {args.gradient_accumulation_steps} = {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"Model / Checkpoint:       {args.model_name_or_path}")
    print(f"Problem Suite:            {args.problem_file}")
    print(f"STaR Iterations:          {args.star_iterations}")
    print(f"Candidates per Problem:   {args.num_candidates} (N={args.num_candidates})")
    print(f"Top-K Accepted / Problem: {args.max_verified_per_problem}")
    print(f"Rollout Backend:          {args.rollout_backend} (dry_run: {args.dry_run})")
    print(f"Output Directory:         {args.output_dir}")
    print("=" * 75)

    problems = load_problem_suite(args.problem_file, max_problems=args.max_problems)
    if getattr(args, "wmdp_cyber", False) or getattr(args, "wmdp_file", None):
        wmdp_source = getattr(args, "wmdp_file", None) or "cais/wmdp"
        wmdp_problems = load_wmdp_cyber_prompts(wmdp_source, max_samples=args.max_problems)
        if wmdp_problems:
            print(f"[✓] WMDP-Cyber benchmark prompts ingested: {len(wmdp_problems)} samples.")
            problems = wmdp_problems + problems

    print(f"Total problems loaded for exploration: {len(problems)}")
    if not problems:
        print("[Error] No problems available for the STaR cycle.")
        return None

    gold_anchors = load_gold_anchors(args.anchor_file, upsample_factor=args.anchor_upsample)
    if gold_anchors:
        print(f"Golden Anchor Samples Loaded: {len(gold_anchors)} (upsample={args.anchor_upsample}x)")

    verifier = STaRVerifier(
        min_think_chars=args.min_think_chars,
        strict_code_syntax=args.strict_code_syntax,
        strict_ground_truth=args.strict_ground_truth,
        enable_prm=getattr(args, "v1_4", False) or getattr(args, "include_prm", False),
        w_prm=getattr(args, "w_prm", 0.25),
    )

    current_model_path = args.model_name_or_path
    current_adapter_path = args.initial_adapter
    iteration_history = []

    rollout_backend = "mock" if args.dry_run else args.rollout_backend

    # ==========================================================================
    # STaR Self-Evolution Loop
    # ==========================================================================
    for iter_idx in range(1, args.star_iterations + 1):
        iter_start = time.time()
        print("\n" + "#" * 75)
        print(f"🔄 STARTING STaR CYCLE - ITERATION {iter_idx}/{args.star_iterations}")
        print("#" * 75)

        # ----------------------------------------------------------------------
        # Step 1: Exploratory Rollouts
        # ----------------------------------------------------------------------
        print(f"\n[1/4] Generating {args.num_candidates} exploratory rollouts per problem via backend '{rollout_backend}'...")
        generator = RolloutGenerator(
            backend=rollout_backend,
            model_name_or_path=current_model_path,
            adapter_path=current_adapter_path,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )

        rollouts = generator.generate_candidates(problems, args.num_candidates)
        print(f"Rollouts completed: {sum(len(r) for r in rollouts)} trajectories generated.")

        # VRAM cleanup
        generator.unload()
        del generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ----------------------------------------------------------------------
        # Step 2: Deterministic Verification & Rejection Sampling
        # ----------------------------------------------------------------------
        print("\n[2/4] Deterministic Verification & Rejection & Pruning...")
        verified_corpus, telemetry = execute_rejection_sampling(
            problems=problems,
            rollouts_by_problem=rollouts,
            verifier=verifier,
            max_verified_per_problem=args.max_verified_per_problem,
        )

        telemetry["iteration"] = iter_idx
        telemetry["elapsed_seconds"] = round(time.time() - iter_start, 2)

        print(f"Global Acceptance Rate:        {telemetry['acceptance_rate']}%")
        print(f"Problems with Approved Solution:{telemetry['accepted_problems_count']}/{len(problems)} ({telemetry['problem_solve_rate']}%)")
        print(f"Approved Trajectories:         {len(verified_corpus)}")
        print(f"Rejections by Format:          {telemetry['rejection_breakdown']['format_failed']}")
        print(f"Rejections by Ground Truth:    {telemetry['rejection_breakdown']['ground_truth_failed']}")
        print(f"Rejections by Syntax Error:    {telemetry['rejection_breakdown']['code_syntax_failed']}")

        if not verified_corpus:
            print(f"[Warning] No approved trajectories in iteration {iter_idx}. Halting self-evolution loop.")
            break

        # ----------------------------------------------------------------------
        # Step 3: Anchor Fusion & Corpus Export
        # ----------------------------------------------------------------------
        print(f"\n[3/4] Assembling Self-Training Corpus for Iteration {iter_idx}...")
        final_iter_dataset = list(verified_corpus)
        if gold_anchors:
            print(f"Merging {len(gold_anchors)} Golden Anchor samples for knowledge retention...")
            final_iter_dataset.extend(gold_anchors)

        telemetry["total_training_samples"] = len(final_iter_dataset)

        iter_corpus_file = os.path.join(args.output_dir, f"star_corpus_iter_{iter_idx}.jsonl")
        with open(iter_corpus_file, "w", encoding="utf-8") as f_out:
            for entry in final_iter_dataset:
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

        iter_telemetry_file = os.path.join(args.output_dir, f"star_telemetry_iter_{iter_idx}.json")
        with open(iter_telemetry_file, "w", encoding="utf-8") as f_tel:
            json.dump(telemetry, f_tel, indent=2, ensure_ascii=False)

        print(f"Iterative STaR corpus saved to: {iter_corpus_file}")
        print(f"Detailed telemetry saved to:    {iter_telemetry_file}")

        # ----------------------------------------------------------------------
        # Step 4: SFT / LoRA Update (New Checkpoint)
        # ----------------------------------------------------------------------
        print(f"\n[4/4] Executing SFT LoRA Update for Iteration {iter_idx}...")
        iter_checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-iter-{iter_idx}")
        updated_adapter_dir = run_sft_update(
            iteration=iter_idx,
            train_file=iter_corpus_file,
            output_checkpoint_dir=iter_checkpoint_dir,
            args=args,
            model_path=current_model_path,
            adapter_path_prev=current_adapter_path if iter_idx == 1 else None,
        )

        current_model_path = updated_adapter_dir
        current_adapter_path = updated_adapter_dir
        iteration_history.append(telemetry)

        print(f"✅ Iteration {iter_idx} completed successfully! New adapter: {current_adapter_path}")

    total_elapsed = round(time.time() - start_total_time, 2)
    summary_file = os.path.join(args.output_dir, "star_pipeline_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f_sum:
        json.dump({
            "total_iterations": len(iteration_history),
            "total_elapsed_seconds": total_elapsed,
            "final_adapter": current_adapter_path,
            "history": iteration_history,
        }, f_sum, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print("🏆 STaR TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total Time:                  {total_elapsed}s")
    print(f"Total Completed Iterations:  {len(iteration_history)}")
    print(f"Final STaR Checkpoint:       {current_adapter_path}")
    print(f"Consolidated Summary at:     {summary_file}")
    print("=" * 75)

    return current_adapter_path


def main():
    args = parse_args()
    run_star_pipeline(args)


if __name__ == "__main__":
    main()
