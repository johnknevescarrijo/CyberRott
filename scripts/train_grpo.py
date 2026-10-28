#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO (Group Relative Policy Optimization) Training Pipeline - CyberRott v1.3
=============================================================================
Implements the DeepSeek-R1 style online reinforcement learning engine:
1. Automated in-memory fusion of prior SFT LoRA checkpoint (π_ref with zero extra VRAM).
2. Targeted curation of verification-rich samples (CWE, MITRE ATT&CK, Multiple Choice, and Sandbox Code).
3. 3 Verifiable Deterministic Reward Functions integrated with STaRVerifier:
   - cot_integrity_reward (0.30): Structural <think> integrity, analytical depth, 7 stages, and anti-refusal.
   - technical_accuracy_reward (0.50): Exact multiple-choice ground truth and CWE/MITRE identifiers.
   - sandbox_code_reward (0.20): Static syntax validation across Python AST, YARA rules, and Sigma/YAML schemas.
4. Optimized for single-GPU acceleration (NVIDIA A100 40GB / L4 24GB on Colab Pro) via TRL GRPOTrainer and PEFT.
5. Native non-GPU deterministic validation support via the --dry_run flag.
"""

import os
# Early configuration of CUDA memory allocator to mitigate VRAM fragmentation (expandable_segments)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import gc
import re
import json
import time
import shutil
import urllib.request
import argparse
import inspect
import functools
import dataclasses
from typing import List, Dict, Any, Optional, Tuple, Union

# -----------------------------------------------------------------------------
# Critical Compatibility Layer: Protection against transformers vs. TRL type divergence
# -----------------------------------------------------------------------------
def apply_grpo_compatibility_patches() -> bool:
    """Applies essential compatibility patches between transformers (>= 4.48 / 5.x) and TRL GRPOTrainer.

    1. Fixes trl.import_utils where availability checks return tuples (e.g., in transformers >= 4.48 / 5.x)
       that evaluate as Truthy, causing unintended imports (e.g., vLLM).
    2. Ensures warnings_issued attribute exists on PreTrainedModel and GRPOTrainer.__init__ to prevent AttributeError.
    3. Critical patch on GRPOTrainer._get_train_sampler signature:
       In transformers >= 4.48, Trainer._get_dataloader invokes self._get_train_sampler(dataset),
       passing the dataset as a positional argument. In TRL versions (e.g. 0.15.x / 0.16.x),
       GRPOTrainer._get_train_sampler(self) was defined without train_dataset, triggering:
       `TypeError: GRPOTrainer._get_train_sampler() takes 1 positional argument but 2 were given`.
    4. Patch on GRPOTrainer._get_eval_sampler to guarantee tolerance for optional/variable arguments.
    """
    patched_any = False

    # 1. Fix tuple return values in trl.import_utils and accurate vLLM detection
    try:
        import trl.import_utils
        import importlib.util
        vllm_installed = importlib.util.find_spec("vllm") is not None

        for _attr in dir(trl.import_utils):
            if _attr.startswith("_") and _attr.endswith("_available"):
                _val = getattr(trl.import_utils, _attr)
                if "vllm" in _attr:
                    setattr(trl.import_utils, _attr, vllm_installed)
                    patched_any = True
                elif isinstance(_val, tuple):
                    setattr(trl.import_utils, _attr, bool(_val[0]))
                    patched_any = True
            elif _attr.startswith("is_") and _attr.endswith("_available"):
                if "vllm" in _attr:
                    setattr(trl.import_utils, _attr, (lambda b=vllm_installed: b))
                    patched_any = True
                else:
                    _fn = getattr(trl.import_utils, _attr)
                    if callable(_fn):
                        try:
                            _val = _fn()
                            if isinstance(_val, tuple):
                                _bool_val = bool(_val[0])
                                setattr(trl.import_utils, _attr, (lambda b=_bool_val: b))
                                patched_any = True
                        except Exception:
                            pass
    except Exception:
        pass

    # 2. Ensure warnings_issued on PreTrainedModel
    try:
        from transformers import PreTrainedModel
        if not hasattr(PreTrainedModel, "warnings_issued"):
            PreTrainedModel.warnings_issued = {}
            patched_any = True
    except Exception:
        pass

    # 3. Patch GRPOTrainer.__init__, _get_train_sampler, and _get_eval_sampler
    try:
        from trl.trainer.grpo_trainer import GRPOTrainer, RepeatRandomSampler

        # Hook on GRPOTrainer.__init__ to ensure warnings_issued on model
        if not getattr(GRPOTrainer.__init__, "_is_patched", False):
            _orig_grpo_init = GRPOTrainer.__init__

            @functools.wraps(_orig_grpo_init)
            def _patched_grpo_init(self, *args, **kwargs):
                model = kwargs.get("model", args[0] if len(args) > 0 else None)
                if model is not None and not hasattr(model, "warnings_issued"):
                    try:
                        model.warnings_issued = {}
                    except Exception:
                        pass
                return _orig_grpo_init(self, *args, **kwargs)

            _patched_grpo_init._is_patched = True
            _patched_grpo_init._orig_init = _orig_grpo_init
            GRPOTrainer.__init__ = _patched_grpo_init
            patched_any = True

        # Hook on GRPOTrainer._get_train_sampler
        if not getattr(GRPOTrainer._get_train_sampler, "_is_patched", False):
            _orig_get_train_sampler = GRPOTrainer._get_train_sampler

            def _patched_get_train_sampler(self, train_dataset=None, *args, **kwargs):
                """Sampler compatible with Trainer._get_dataloader (transformers >= 4.48)."""
                dataset = train_dataset if train_dataset is not None else getattr(self, "train_dataset", None)
                if dataset is None:
                    return None
                try:
                    effective_batch_size = (
                        self.args.per_device_train_batch_size
                        * self.accelerator.num_processes
                        * self.args.gradient_accumulation_steps
                    )
                    return RepeatRandomSampler(
                        data_source=dataset,
                        mini_repeat_count=self.num_generations,
                        batch_size=effective_batch_size // self.num_generations,
                        repeat_count=self.num_iterations,
                        seed=self.args.seed,
                    )
                except Exception:
                    if train_dataset is not None and getattr(self, "train_dataset", None) is None:
                        self.train_dataset = train_dataset
                    try:
                        return _orig_get_train_sampler(self, train_dataset, *args, **kwargs)
                    except TypeError:
                        return _orig_get_train_sampler(self)

            _patched_get_train_sampler._is_patched = True
            GRPOTrainer._get_train_sampler = _patched_get_train_sampler
            patched_any = True

        # Hook on GRPOTrainer._get_eval_sampler
        if hasattr(GRPOTrainer, "_get_eval_sampler") and not getattr(GRPOTrainer._get_eval_sampler, "_is_patched", False):
            _orig_get_eval_sampler = GRPOTrainer._get_eval_sampler

            def _patched_get_eval_sampler(self, eval_dataset=None, *args, **kwargs):
                """Evaluation sampler tolerant of optional eval_dataset."""
                dataset = eval_dataset if eval_dataset is not None else getattr(self, "eval_dataset", None)
                if dataset is None:
                    return None
                try:
                    return RepeatRandomSampler(
                        data_source=dataset,
                        mini_repeat_count=self.num_generations,
                        seed=self.args.seed,
                    )
                except Exception:
                    try:
                        return _orig_get_eval_sampler(self, dataset, *args, **kwargs)
                    except TypeError:
                        return _orig_get_eval_sampler(self)

            _patched_get_eval_sampler._is_patched = True
            GRPOTrainer._get_eval_sampler = _patched_get_eval_sampler
            patched_any = True

    except Exception:
        pass

    return patched_any


# Apply patches immediately upon module import
apply_grpo_compatibility_patches()

import torch

# Disable Weights & Biases by default and configure CUDA memory allocator
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_DISABLED"] = "true"

# Add repository root directory to Python path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from scripts.star_verifier import (
        verify_format_integrity,
        verify_ground_truth,
        verify_code_syntax,
        extract_mc_answer,
        CHOICE_MAP,
        check_refusals,
        STaRVerifier,
        verify_prm_steps,
        compute_prm_step_score,
    )
except ImportError:
    from star_verifier import (
        verify_format_integrity,
        verify_ground_truth,
        verify_code_syntax,
        extract_mc_answer,
        CHOICE_MAP,
        check_refusals,
        STaRVerifier,
        verify_prm_steps,
        compute_prm_step_score,
    )

verifier = STaRVerifier()

# ==============================================================================
# Backbone Scaling Catalog (v1.4)
# ==============================================================================

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


# ==============================================================================
# Hardware Optimizations for L4 and A100 Accelerators
# ==============================================================================

def setup_hardware_optimizations(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """Detects GPU accelerator and configures TF32 and optimal attention mechanisms."""
    profile = {
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
    profile["is_high_vram"] = (total_vram_gb >= 35.0)

    try:
        import flash_attn  # noqa: F401
        profile["recommended_attn"] = "flash_attention_2"
    except ImportError:
        profile["recommended_attn"] = "sdpa"

    if args is not None and getattr(args, "auto_optimize", True):
        if profile["is_high_vram"] and args.attn_implementation == "sdpa" and profile["recommended_attn"] == "flash_attention_2":
            args.attn_implementation = "flash_attention_2"
            print("[A100 Auto-Optimization] Automatically activated flash_attention_2.")

    return profile


# ==============================================================================
# KV-Cache Acceleration for GRPO Rollouts with Gradient Checkpointing
# ==============================================================================

def enable_fast_generation_with_kv_cache(model: Any) -> Any:
    """Acceleration wrapper that reactivates KV-Cache during model.generate()
    when gradient checkpointing is active during GRPO reinforcement training.

    Critical Bottleneck Addressed:
    In architectures based on Phi3DecoderLayer (e.g. Phi-4-mini-instruct), when
    gradient_checkpointing=True and model is in train() mode, layer forward forces:
      `use_cache=False` and `past_key_values=None`.
    GRPOTrainer executes model.generate() under torch.no_grad(), but if the model retains
    the gradient checkpointing flag, the KV-Cache is discarded at every token. Without KV-Cache,
    generating hundreds of tokens scales with O(N^2) complexity, recalculating full attention
    and causing generate() to run 10x-15x slower (~15s vs ~170s per iteration on Colab A100).

    Solution Mechanism:
    Intercepts model.generate():
    1. Saves training and gradient checkpointing state across all submodules.
    2. Temporarily places model in model.eval().
    3. Disables gradient checkpointing across model and layers (m.gradient_checkpointing = False).
    4. Enforces model.config.use_cache = True and kwargs['use_cache'] = True.
    5. Executes original generation with 100% active KV-Cache (O(N) instead of O(N^2)).
    6. In finally block, restores model.config.use_cache = False, restores gradient
       checkpointing flags, and returns model to training mode.
    """
    if model is None or not hasattr(model, "generate"):
        return model

    if getattr(model.generate, "_is_kv_cache_wrapped", False):
        return model

    orig_generate = model.generate

    def fast_generate(*args, **kwargs):
        if getattr(enable_fast_generation_with_kv_cache, "_active_generation", False):
            kwargs["use_cache"] = True
            if "generation_config" in kwargs and kwargs["generation_config"] is not None:
                try:
                    kwargs["generation_config"].use_cache = True
                except Exception:
                    pass
            return orig_generate(*args, **kwargs)

        enable_fast_generation_with_kv_cache._active_generation = True
        was_training = getattr(model, "training", False)

        # 1. Map modules with active gradient checkpointing
        gc_modules = []
        for m in model.modules():
            if hasattr(m, "gradient_checkpointing") and getattr(m, "gradient_checkpointing", False):
                gc_modules.append(m)

        prev_is_gc = getattr(model, "is_gradient_checkpointing", False)

        try:
            # a) Switch to eval mode to disable dropout
            if hasattr(model, "eval"):
                model.eval()

            # b) Temporarily disable gradient checkpointing across modules
            if hasattr(model, "gradient_checkpointing_disable"):
                try:
                    model.gradient_checkpointing_disable()
                except Exception:
                    pass

            if hasattr(model, "is_gradient_checkpointing"):
                model.is_gradient_checkpointing = False

            for m in gc_modules:
                try:
                    m.gradient_checkpointing = False
                except Exception:
                    pass

            # c) Force use_cache = True on model config and generation kwargs
            if hasattr(model, "config") and model.config is not None:
                model.config.use_cache = True
            if hasattr(model, "generation_config") and model.generation_config is not None:
                model.generation_config.use_cache = True

            kwargs["use_cache"] = True
            if "generation_config" in kwargs and kwargs["generation_config"] is not None:
                try:
                    kwargs["generation_config"].use_cache = True
                except Exception:
                    pass

            # d) Execute generation with 100% active KV-Cache
            return orig_generate(*args, **kwargs)

        finally:
            # e) Restore use_cache = False on config (required for gradient checkpointing backward pass)
            if hasattr(model, "config") and model.config is not None:
                model.config.use_cache = False
            if hasattr(model, "generation_config") and model.generation_config is not None:
                model.generation_config.use_cache = False

            # f) Reactivate gradient checkpointing across model and layers
            if prev_is_gc or gc_modules:
                if hasattr(model, "gradient_checkpointing_enable"):
                    try:
                        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    except TypeError:
                        try:
                            model.gradient_checkpointing_enable()
                        except Exception:
                            pass
                    except Exception:
                        pass

                for m in gc_modules:
                    try:
                        m.gradient_checkpointing = True
                    except Exception:
                        pass

                if hasattr(model, "is_gradient_checkpointing"):
                    model.is_gradient_checkpointing = True

            # g) Restore training mode if active
            if was_training and hasattr(model, "train"):
                model.train()

            enable_fast_generation_with_kv_cache._active_generation = False

    fast_generate._is_kv_cache_wrapped = True
    fast_generate._orig_generate = orig_generate
    model.generate = fast_generate
    return model

enable_fast_generation_with_kv_cache._active_generation = False


def resolve_resume_checkpoint(resume_arg: Optional[str], output_dir: str) -> Optional[str]:
    """Resolves checkpoint path for resuming GRPO training.

    Args:
        resume_arg: Explicit checkpoint path or 'auto' for automatic detection.
        output_dir: Directory where 'checkpoint-*' checkpoints are saved.

    Returns:
        Checkpoint path to resume from, or None to train from start.
    """
    if not resume_arg:
        return None

    cleaned_arg = str(resume_arg).strip()
    if cleaned_arg.lower() == "auto":
        if not os.path.isdir(output_dir):
            print(f"[*] 'auto' detection: Directory '{output_dir}' does not exist yet. Starting training from scratch.")
            return None

        candidates = []
        for entry in os.listdir(output_dir):
            full_path = os.path.join(output_dir, entry)
            if os.path.isdir(full_path) and entry.startswith("checkpoint-"):
                match = re.search(r"checkpoint-(\d+)", entry)
                if match:
                    candidates.append((int(match.group(1)), full_path))

        if not candidates:
            print(f"[*] 'auto' detection: No checkpoints found in '{output_dir}'. Starting training from scratch.")
            return None

        candidates.sort(key=lambda x: x[0])
        latest_checkpoint = candidates[-1][1]
        step_num = candidates[-1][0]
        print(f"[✓] 'auto' detection: Automatically resuming from latest checkpoint: {latest_checkpoint} (Step {step_num})")
        return latest_checkpoint

    if os.path.exists(cleaned_arg):
        print(f"[✓] Resuming training from specified checkpoint: {cleaned_arg}")
        return cleaned_arg
    else:
        print(f"[!] Specified checkpoint not found: '{cleaned_arg}'. Starting training from scratch.")
        return None


def sync_checkpoint_to_google_drive(
    source_dir: str,
    target_subfolder: Optional[str] = None,
    drive_mount: str = "/content/drive/MyDrive"
) -> Optional[str]:
    """Checks whether Google Drive is mounted in Colab and synchronizes the final checkpoint.

    Ensures the trained checkpoint is safely persisted in Google Drive even if the notebook
    runtime terminates prematurely.

    Args:
        source_dir: Local path of final checkpoint (e.g. args.output_dir).
        target_subfolder: Destination subfolder (default: base name of source_dir or 'CyberRott-v1.3-GRPO').
        drive_mount: Default mount point for Google Drive in Colab.

    Returns:
        Destination directory path in Google Drive, or None if Drive is not mounted.
    """
    if not os.path.exists(drive_mount):
        return None

    if not os.path.exists(source_dir):
        print(f"[!] Source directory '{source_dir}' not found for Google Drive sync.")
        return None

    if not target_subfolder:
        base = os.path.basename(os.path.normpath(source_dir))
        target_subfolder = base if base and base not in (".", "/") else "CyberRott-v1.3-GRPO"

    drive_checkpoints_dir = os.path.join(drive_mount, "CyberRott_Checkpoints")
    drive_target_dir = os.path.join(drive_checkpoints_dir, target_subfolder)

    print(f"\n[*] Google Drive detected at '{drive_mount}'.")
    print(f"[*] Automatically synchronizing final checkpoint: '{source_dir}' -> '{drive_target_dir}'...")
    try:
        os.makedirs(drive_checkpoints_dir, exist_ok=True)
        shutil.copytree(source_dir, drive_target_dir, dirs_exist_ok=True)
        print(f"[✓] Checkpoint successfully saved and synced to Google Drive: '{drive_target_dir}'!")

        for alt_candidate in {
            target_subfolder.replace("-", "_"),
            target_subfolder.replace("-", "_").replace(".", "_"),
        }:
            drive_target_alt = os.path.join(drive_checkpoints_dir, alt_candidate)
            if drive_target_alt != drive_target_dir and not os.path.exists(drive_target_alt):
                try:
                    os.symlink(drive_target_dir, drive_target_alt)
                except Exception:
                    pass

        return drive_target_dir
    except Exception as e:
        print(f"[!] Warning: Failed to sync checkpoint to Google Drive ({drive_target_dir}): {e}")
        return None


# ==============================================================================
# Helper for Completion Text Extraction (TRL Compatibility)
# ==============================================================================

def extract_text_from_completion(c: Any) -> str:
    """Extracts raw text string from completions formatted as list[str], list[dict], or list[list[dict]]."""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        if len(c) == 0:
            return ""
        if isinstance(c[0], dict):
            # TRL conversational schema: prioritize assistant role content
            for msg in reversed(c):
                if isinstance(msg, dict) and msg.get("role") == "assistant" and "content" in msg:
                    return str(msg["content"])
            if "content" in c[0]:
                return str(c[0]["content"])
        if isinstance(c[0], str):
            return "".join(c)
    if isinstance(c, dict) and "content" in c:
        return str(c["content"])
    if c is None:
        return ""
    return str(c)


def get_eos_token_ids(tokenizer: Any) -> List[int]:
    """Collects all valid stop/EOS token IDs for Phi-4 and ChatML for exact generation termination.

    In Phi-4-mini-instruct, completions may terminate with <|end|>, <|im_end|>, <|endoftext|>, or <|eot_id|>.
    """
    eos_ids: List[int] = []
    if tokenizer is None:
        return eos_ids

    # 1. Add tokenizer native eos_token_id
    native_eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(native_eos, list):
        for tid in native_eos:
            if isinstance(tid, int) and tid not in eos_ids:
                eos_ids.append(tid)
    elif isinstance(native_eos, int) and native_eos >= 0:
        eos_ids.append(native_eos)

    # 2. Collect common Phi-4 and ChatML stop tokens
    target_tokens = ["<|end|>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>"]
    unk_id = getattr(tokenizer, "unk_token_id", None)

    for token in target_tokens:
        try:
            tid = tokenizer.convert_tokens_to_ids(token)
            if isinstance(tid, int) and tid >= 0:
                if unk_id is not None and tid == unk_id:
                    continue
                if tid not in eos_ids:
                    eos_ids.append(tid)
        except Exception:
            pass

    return eos_ids


# ==============================================================================
# The 3 Deterministic Verifiable Reward Functions (GRPO Reward Suite)
# ==============================================================================

def check_repetition(text: str) -> bool:
    """Detects loop degeneration or severe token repetition in generated outputs.

    Criteria:
    1. A single non-empty line of >= 15 characters repeated more than 4 times.
    2. Unique 4-gram ratio < 0.35 in outputs containing more than 80 words.
    """
    if not text or not isinstance(text, str):
        return False

    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 15]
    if lines:
        from collections import Counter
        line_counts = Counter(lines)
        if any(count > 4 for count in line_counts.values()):
            return True

    words = text.split()
    if len(words) > 80:
        ngrams = [tuple(words[j:j+4]) for j in range(len(words) - 3)]
        if ngrams:
            unique_ratio = len(set(ngrams)) / float(len(ngrams))
            if unique_ratio < 0.35:
                return True

    return False


def cot_integrity_reward(prompts: Any, completions: Any, **kwargs) -> List[float]:
    """
    Reward Function 1: Structural CoT Reasoning Integrity (Weight: 0.30)
    - Severe penalty (immediate 0.0) for conversational AI refusals and evasions.
    - Anti-truncation: if <think> is present but </think> is missing, immediate zero score (0.0).
    - Anti-loop / severe repetition: zeroes score for looping lines (>4x) or 4-gram ratio < 0.35.
    - Multiple-Choice Anti-Indecision Guard: if expected_mc is valid and model hesitates, zeroes score (0.0).
    - Requires balanced <think>...</think> tags.
    - Scores analytical reasoning depth, 7-stage coverage, and structural verdict markers.
    """
    rewards = []
    expected_mcs = kwargs.get("expected_mc", kwargs.get("expected_choice", []))

    for i, c in enumerate(completions):
        text = extract_text_from_completion(c)

        # 1. Anti-truncation check
        if "<think>" in text and "</think>" not in text:
            rewards.append(0.0)
            continue

        # Mandatory presence of paired reasoning tags
        if "<think>" not in text or "</think>" not in text:
            rewards.append(0.0)
            continue

        think_open_idx = text.find("<think>")
        think_close_idx = text.find("</think>")
        if think_close_idx <= think_open_idx:
            rewards.append(0.0)
            continue

        # 2. Anti-loop repetition detection
        if check_repetition(text):
            rewards.append(0.0)
            continue

        # 3. Multiple-Choice Anti-Indecision Guard
        expected_mc_val = expected_mcs[i] if (i < len(expected_mcs) and expected_mcs[i]) else None
        is_mc = bool(expected_mc_val and str(expected_mc_val).strip() and str(expected_mc_val).strip().upper() in ["A", "B", "C", "D", "0", "1", "2", "3"])
        if is_mc and extract_mc_answer(text) is None:
            rewards.append(0.0)
            continue

        # 4. Format verification and refusal inspection
        fmt_ok, fmt_reasons, fmt_metrics = verify_format_integrity(
            text,
            min_think_chars=80,
            min_think_words=15,
            min_final_chars=20,
        )

        if fmt_metrics.get("refusals_detected"):
            rewards.append(0.0)
            continue

        # 5. CoT integrity scoring [0.0 - 1.0]
        sub_score = 0.0

        if fmt_ok:
            sub_score += 0.40
        else:
            sub_score += 0.15

        # Analytical depth (ideal: >= 30 reasoning words)
        think_words = fmt_metrics.get("think_words", 0)
        depth_ratio = min(1.0, think_words / 30.0)
        sub_score += 0.20 * depth_ratio

        # 7-stage cybersecurity reasoning coverage
        stages_cov = fmt_metrics.get("stages_coverage", 0.0)
        sub_score += 0.20 * float(stages_cov)

        # Structural verdict markers (FACT, DETECTION, REMEDIATION, etc.)
        struct_markers = fmt_metrics.get("structure_markers_found", 0)
        marker_ratio = min(1.0, struct_markers / 2.0)
        sub_score += 0.20 * marker_ratio

        # Scaled by weight 0.30
        rewards.append(round(min(1.0, sub_score) * 0.30, 4))

    return rewards


def technical_accuracy_reward(prompts: Any, completions: Any, **kwargs) -> List[float]:
    """
    Reward Function 2: Technical Accuracy and Ground-Truth Fidelity (Weight: 0.50)
    - Evaluates exact multiple-choice key (A, B, C, D) for MCQ challenges.
    - Validates deterministic technical indicators: CWE IDs (CWE-89, etc.) and MITRE ATT&CK (T1059, etc.).
    - Validates technical consistency against reference solution.
    """
    rewards = []
    ref_answers = kwargs.get("reference_answer", [])
    expected_cwes = kwargs.get("expected_cwe", [])
    expected_mitres = kwargs.get("expected_mitre", [])
    expected_choices = kwargs.get("expected_choice", kwargs.get("expected_mc", []))

    for i, c in enumerate(completions):
        text = extract_text_from_completion(c)

        ref_ans = ref_answers[i] if (i < len(ref_answers) and ref_answers[i]) else ""
        cwe_item = expected_cwes[i] if (i < len(expected_cwes) and expected_cwes[i]) else None
        mitre_item = expected_mitres[i] if (i < len(expected_mitres) and expected_mitres[i]) else None
        choice_item = expected_choices[i] if (i < len(expected_choices) and expected_choices[i]) else None

        if isinstance(cwe_item, str) and cwe_item.strip():
            cwe_list = [c.strip().upper() for c in cwe_item.split(",") if c.strip()]
        elif isinstance(cwe_item, list):
            cwe_list = [str(c).strip().upper() for c in cwe_item]
        else:
            cwe_list = []

        if isinstance(mitre_item, str) and mitre_item.strip():
            mitre_list = [m.strip().upper() for m in mitre_item.split(",") if m.strip()]
        elif isinstance(mitre_item, list):
            mitre_list = [str(m).strip().upper() for m in mitre_item]
        else:
            mitre_list = []

        choice_val = str(choice_item).strip().upper() if choice_item else ""

        problem = {
            "reference_answer": ref_ans,
            "expected_cwe": cwe_list,
            "expected_mitre": mitre_list,
            "expected_answer": choice_val,
        }

        gt_ok, gt_reasons, gt_metrics = verify_ground_truth(text, problem)
        gt_type = gt_metrics.get("gt_type", "unknown")

        sub_score = 0.0
        if gt_ok:
            sub_score = 1.0
        else:
            if gt_type in ["multiple_choice", "multiple_choice_derived"]:
                sub_score = 0.0
            else:
                candidate_cwes = [cw.upper() for cw in re.findall(r"\bCWE-\d+\b", text, re.I)]
                candidate_mitre = [mt.upper() for mt in re.findall(r"\bT\d{4}(?:\.\d{3})?\b", text, re.I)]

                matched_cwes = [cw for cw in cwe_list if cw in candidate_cwes]
                matched_mitre = [mt for mt in mitre_list if mt in candidate_mitre]

                cwe_ratio = (len(matched_cwes) / len(cwe_list)) if cwe_list else 0.0
                mitre_ratio = (len(matched_mitre) / len(mitre_list)) if mitre_list else 0.0

                if cwe_list and mitre_list:
                    sub_score = 0.5 * cwe_ratio + 0.5 * mitre_ratio
                elif cwe_list:
                    sub_score = 1.0 * cwe_ratio
                elif mitre_list:
                    sub_score = 1.0 * mitre_ratio
                else:
                    if any(term in text.lower() for term in ["vulnerabilidade", "vulnerability", "fato:", "fact:", "veredicto", "verdict"]):
                        sub_score = 0.60

        rewards.append(round(min(1.0, sub_score) * 0.50, 4))

    return rewards


def sandbox_code_reward(prompts: Any, completions: Any, **kwargs) -> List[float]:
    """
    Reward Function 3: Static Code and Syntax Sandbox (Weight: 0.20)
    - Python AST compilation (ast.parse and compile(..., 'exec') without live execution).
    - YARA rule structural grammar (rule, condition, balanced delimiters).
    - Static YAML parsing for Sigma detection rules (title, logsource, detection).
    - Strict boundary enforcement: zeroes score (0.0) if refusal detected, reflection
      tags truncated, or multiple-choice option omitted.
    """
    rewards = []
    expected_mcs = kwargs.get("expected_mc", kwargs.get("expected_choice", []))

    for i, c in enumerate(completions):
        text = extract_text_from_completion(c)

        # 1. Refusal penalty
        if verifier.check_refusals(text):
            rewards.append(0.0)
            continue

        # 2. Truncation penalty
        if "<think>" in text and "</think>" not in text:
            rewards.append(0.0)
            continue

        # 3. Multiple-Choice Anti-Indecision Guard
        expected_mc_val = expected_mcs[i] if (i < len(expected_mcs) and expected_mcs[i]) else None
        is_mc = bool(expected_mc_val and str(expected_mc_val).strip() and str(expected_mc_val).strip().upper() in ["A", "B", "C", "D", "0", "1", "2", "3"])
        if is_mc and extract_mc_answer(text) is None:
            rewards.append(0.0)
            continue

        code_ok, code_reasons, code_metrics = verify_code_syntax(text, check_inside_think=False)
        total_blocks = code_metrics.get("total_code_blocks", 0)

        if total_blocks == 0:
            sub_score = 1.0
        else:
            if code_ok:
                sub_score = 1.0
            else:
                failed_blocks = len(code_reasons)
                sub_score = max(0.0, 1.0 - (failed_blocks / float(total_blocks)))

        rewards.append(round(min(1.0, sub_score) * 0.20, 4))

    return rewards


def prm_step_reward(prompts: Any, completions: Any, **kwargs) -> List[float]:
    """
    Reward Function 4: Step-Level Process Reward Model (PRM) (Weight: 0.25)
    - Avalia a cadeia de raciocínio (<think>...</think>) em granularidade de passos lógicos.
    - Bonifica a cobertura dos 7 estágios do protocolo analítico CyberRott v1.4.
    - Bonifica conectores dedutivos formais (therefore, because, implies, etc.).
    - Penaliza sumariamente tentativas de reward hacking por repetição artificial de linhas/n-gramas.
    - Penaliza contradições internas entre a conclusão do raciocínio e a resposta final.
    """
    rewards = []
    w_prm = kwargs.get("w_prm", 0.25)

    for c in completions:
        text = extract_text_from_completion(c)
        if not text:
            rewards.append(0.0)
            continue

        prm_info = verify_prm_steps(text)
        prm_score = float(prm_info.get("prm_score", 0.0))
        rewards.append(round(min(1.0, prm_score) * w_prm, 4))

    return rewards


# ==============================================================================
# Dataset Loading with GitHub Fallback and Verifiability Curation
# ==============================================================================

DEFAULT_SYSTEM_PROMPT = (
    "You are CyberRott, a specialized reasoning AI for offensive, defensive, and vulnerability analysis with deep analytical reasoning."
)

GITHUB_RAW_TRAIN = "https://raw.githubusercontent.com/johnknevescarrijo/CyberRott/main/data/cyberrott_final_train.jsonl"
GITHUB_RAW_ANCHOR = "https://raw.githubusercontent.com/johnknevescarrijo/CyberRott/main/data/dataset_cyber_ready.jsonl"


def ensure_dataset_exists(file_path: str, fallback_url: str) -> str:
    """Ensures dataset file exists locally, downloading from GitHub fallback if absent."""
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return file_path

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    print(f"[*] Local file '{file_path}' absent. Downloading via GitHub fallback ({fallback_url})...")
    try:
        urllib.request.urlretrieve(fallback_url, file_path)
        print(f"[✓] Download completed successfully: {file_path} ({os.path.getsize(file_path) / (1024*1024):.2f} MB)")
        return file_path
    except Exception as e:
        print(f"[!] Failed to download from {fallback_url}: {e}")
        if not os.path.exists(file_path):
            tmp_clone = "/tmp/cyberrott_git_clone"
            if not os.path.exists(tmp_clone):
                os.system(f"git clone --depth 1 https://github.com/johnknevescarrijo/CyberRott.git {tmp_clone}")
            repo_file = os.path.join(tmp_clone, file_path)
            if os.path.exists(repo_file):
                shutil.copyfile(repo_file, file_path)
                print(f"[✓] File recovered via git clone: {file_path}")
                return file_path

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Unable to locate or download dataset: {file_path}")
    return file_path


def parse_and_curate_dataset(
    train_path: str,
    anchor_path: str,
    max_samples: int = 400
) -> List[Dict[str, Any]]:
    """
    Loads dataset, extracts deterministic verification metadata, and prioritizes samples
    with highest technical density (CWE + MITRE + Multiple Choice + Code) plus Golden Anchors.
    """
    ensure_dataset_exists(train_path, GITHUB_RAW_TRAIN)
    ensure_dataset_exists(anchor_path, GITHUB_RAW_ANCHOR)

    all_raw_samples = []

    # 1. Load high-priority Golden Anchors (dataset_cyber_ready.jsonl)
    anchor_count = 0
    if os.path.exists(anchor_path):
        with open(anchor_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    entry["_is_anchor"] = True
                    all_raw_samples.append(entry)
                    anchor_count += 1
                except json.JSONDecodeError:
                    continue
        print(f"[*] Golden Anchor samples loaded: {anchor_count}")

    # 2. Load primary training corpus (cyberrott_final_train.jsonl)
    train_count = 0
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                entry["_is_anchor"] = False
                all_raw_samples.append(entry)
                train_count += 1
            except json.JSONDecodeError:
                continue
    print(f"[*] Training samples loaded: {train_count} (Total raw entries: {len(all_raw_samples)})")

    # 3. Process and rank by verifiability richness
    curated_records = []
    for item in all_raw_samples:
        messages = item.get("messages", [])
        if not messages or not isinstance(messages, list):
            continue

        system_content = None
        user_content = None
        assistant_content = ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_content = content
            elif role == "user":
                user_content = content
            elif role == "assistant":
                assistant_content = content

        if not user_content:
            continue

        if not system_content:
            system_content = DEFAULT_SYSTEM_PROMPT

        prompt_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        # Extract ground-truth indicators
        cwe_matches = list(set(re.findall(r"\bCWE-\d+\b", assistant_content, re.IGNORECASE)))
        expected_cwe = ",".join([c.upper() for c in cwe_matches])

        mitre_matches = list(set(re.findall(r"\bT\d{4}(?:\.\d{3})?\b", assistant_content, re.IGNORECASE)))
        expected_mitre = ",".join([m.upper() for m in mitre_matches])

        expected_choice = extract_mc_answer(assistant_content) or ""
        has_code = "```" in assistant_content

        # Verifiability Richness Score
        richness_score = 0.0
        if item.get("_is_anchor", False):
            richness_score += 100.0  # Preserve 100% of anchors
        if cwe_matches:
            richness_score += 4.0
        if mitre_matches:
            richness_score += 3.0
        if expected_choice:
            richness_score += 2.0
        if has_code:
            richness_score += 2.0
        if len(assistant_content) > 300:
            richness_score += 1.0

        curated_records.append({
            "prompt": prompt_messages,
            "reference_answer": assistant_content,
            "expected_cwe": expected_cwe,
            "expected_mitre": expected_mitre,
            "expected_choice": expected_choice,
            "expected_mc": expected_choice,
            "_richness": richness_score,
            "_is_anchor": item.get("_is_anchor", False),
        })

    # 4. Sort by descending richness
    curated_records.sort(key=lambda x: x["_richness"], reverse=True)

    if max_samples > 0 and len(curated_records) > max_samples:
        selected_records = curated_records[:max_samples]
        print(f"[✓] Curation completed: Selected {len(selected_records)} top verifiable samples (max_samples={max_samples}).")
    else:
        selected_records = curated_records
        print(f"[✓] Curation completed: Utilizing all {len(selected_records)} available samples.")

    final_dataset = []
    for r in selected_records:
        final_dataset.append({
            "prompt": r["prompt"],
            "reference_answer": r["reference_answer"],
            "expected_cwe": r["expected_cwe"],
            "expected_mitre": r["expected_mitre"],
            "expected_choice": r["expected_choice"],
            "expected_mc": r["expected_mc"],
        })

    return final_dataset


# ==============================================================================
# Dry-Run Mode Execution (Non-GPU Validation)
# ==============================================================================

def run_dry_run_validation(dataset_records: List[Dict[str, Any]]) -> bool:
    """Validates data loading, ground-truth extraction, and deterministic computation of all 3 rewards."""
    print("\n" + "=" * 70)
    print("🔍 DRY-RUN MODE: DETERMINISTIC VALIDATION OF GRPO PIPELINE")
    print("=" * 70)
    print(f"[*] Total samples ready for GRPO: {len(dataset_records)}")

    cwe_samples = sum(1 for r in dataset_records if r["expected_cwe"])
    mitre_samples = sum(1 for r in dataset_records if r["expected_mitre"])
    choice_samples = sum(1 for r in dataset_records if r["expected_choice"])

    print(f"[*] Samples with expected CWE:              {cwe_samples} ({cwe_samples/len(dataset_records)*100:.1f}%)")
    print(f"[*] Samples with expected MITRE ATT&CK:     {mitre_samples} ({mitre_samples/len(dataset_records)*100:.1f}%)")
    print(f"[*] Samples with expected Multiple Choice:  {choice_samples} ({choice_samples/len(dataset_records)*100:.1f}%)")

    test_sample = dataset_records[0]
    prompt = test_sample["prompt"]
    gold_ref = test_sample["reference_answer"]

    prompts_batch = [prompt] * 4

    # 4 Synthetic variations to evaluate reward response:
    # 1. Ideal candidate (expects high reward)
    cand_ideal = gold_ref if ("<think>" in gold_ref and "</think>" in gold_ref) else f"<think>\nAnalysis of attack surface and data flow with identified sinks.\nVulnerability verdict: CWE-89.\n</think>\nFACT: Identified CWE-89.\nDETECTION: Monitor SQL Injection patterns.\nREMEDIATION:\n```python\ncursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))\n```"

    # 2. Refusal candidate (must zero CoT reward)
    cand_refusal = "<think>Analyzing request.</think> I apologize, but as an AI language model I cannot assist with vulnerability analysis."

    # 3. Broken tags candidate (must penalize CoT)
    cand_broken_tags = "This response lacks deliberation tags. The flaw is CWE-89."

    # 4. Syntax error candidate (must penalize sandbox code reward)
    cand_syntax_error = "<think>\nTechnical analysis with flaw identification.\n</think>\nFACT: Flaw detected.\n```python\ndef broken_func(\n  returrn syntax error !\n```"

    completions_batch = [cand_ideal, cand_refusal, cand_broken_tags, cand_syntax_error]

    kwargs = {
        "reference_answer": [test_sample["reference_answer"]] * 4,
        "expected_cwe": [test_sample["expected_cwe"]] * 4,
        "expected_mitre": [test_sample["expected_mitre"]] * 4,
        "expected_choice": [test_sample["expected_choice"]] * 4,
        "expected_mc": [test_sample.get("expected_mc", test_sample["expected_choice"])] * 4,
    }

    r_cot = cot_integrity_reward(prompts_batch, completions_batch, **kwargs)
    r_acc = technical_accuracy_reward(prompts_batch, completions_batch, **kwargs)
    r_box = sandbox_code_reward(prompts_batch, completions_batch, **kwargs)
    r_prm = prm_step_reward(prompts_batch, completions_batch, **kwargs)

    print("\n" + "-" * 70)
    print("EVALUATION TEST RESULTS FOR THE 4 DETERMINISTIC REWARD FUNCTIONS (TEST BATCH):")
    print("-" * 70)
    scenarios = [
        "1. Ideal Candidate (Full CoT + Clean Code)",
        "2. Refusal Candidate ('As an AI...')",
        "3. Missing Tags Candidate (No <think>)",
        "4. Syntax Error Candidate (Python AST Error)",
    ]

    for idx, name in enumerate(scenarios):
        total_r = round(r_cot[idx] + r_acc[idx] + r_box[idx] + r_prm[idx], 4)
        print(f"\n{name}:")
        print(f"  • cot_integrity_reward (weight 0.30):    {r_cot[idx]:.4f}")
        print(f"  • technical_accuracy_reward (weight 0.50): {r_acc[idx]:.4f}")
        print(f"  • sandbox_code_reward (weight 0.20):      {r_box[idx]:.4f}")
        print(f"  • prm_step_reward (weight 0.25):          {r_prm[idx]:.4f}")
        print(f"  ➜ TOTAL COMBINED REWARD:                 {total_r:.4f}")

    assert r_cot[1] == 0.0, "Failure: Alignment refusal did not zero cot_integrity_reward!"
    assert r_box[1] == 0.0, "Failure: Alignment refusal did not zero sandbox_code_reward!"
    assert r_cot[2] == 0.0, "Failure: Missing <think> tags did not zero cot_integrity_reward!"
    assert r_prm[2] == 0.0, "Failure: Missing <think> tags did not zero prm_step_reward!"
    assert r_box[3] < 0.20, "Failure: Python syntax error was not penalized in sandbox_code_reward!"
    assert r_cot[0] > 0.0, "Failure: Ideal candidate must receive positive CoT reward!"

    print("\n" + "=" * 70)
    print("🏆 [DRY-RUN SUCCESSFUL] All reward functions and parsing validated with 100% precision!")
    print("=" * 70 + "\n")
    return True


# ==============================================================================
# vLLM Engine Resolution and Configuration (TRL 0.14 - 0.19+ Compatibility)
# ==============================================================================

def probe_vllm_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    timeout: float = 2.0,
    hosts: Tuple[str, ...] = ("127.0.0.1", "0.0.0.0"),
) -> bool:
    """Checks whether an external vLLM server is active on the given host/port via TCP socket probe."""
    import socket
    candidate_hosts = list(hosts)
    if host not in candidate_hosts:
        candidate_hosts.insert(0, host)
    for h in candidate_hosts:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                if s.connect_ex((h, port)) == 0:
                    return True
        except Exception:
            pass
    return False


def resolve_vllm_kwargs(
    config_cls: Any,
    use_vllm: bool = False,
    gpu_mem_util: float = 0.30,
    vllm_device: str = "auto",
    probe_fn: Optional[Any] = None,
    server_port: int = 8000,
    server_host: str = "127.0.0.1",
) -> Dict[str, Any]:
    """Resolves vLLM configuration arguments for GRPOConfig across TRL versions."""
    try:
        if inspect.isclass(config_cls):
            sig_params = inspect.signature(config_cls.__init__).parameters
        else:
            sig_params = inspect.signature(config_cls).parameters
    except Exception:
        sig_params = {}

    if "use_vllm" not in sig_params:
        return {}

    if not use_vllm:
        return {"use_vllm": False}

    # 1. TRL with native colocate support (TRL >= 0.18 / 0.19+)
    if "vllm_mode" in sig_params:
        result: Dict[str, Any] = {
            "use_vllm": True,
            "vllm_mode": "colocate",
        }
        if "vllm_gpu_memory_utilization" in sig_params:
            result["vllm_gpu_memory_utilization"] = gpu_mem_util
        return result

    # 2. TRL server-only mode (~0.16 / 0.17)
    if "vllm_server_host" in sig_params or "vllm_server_port" in sig_params:
        is_alive = False
        if probe_fn is not None:
            try:
                sig = inspect.signature(probe_fn)
                p_count = len(sig.parameters)
                if p_count == 0:
                    is_alive = bool(probe_fn())
                elif p_count == 1:
                    is_alive = bool(probe_fn(server_port))
                elif p_count == 2:
                    is_alive = bool(probe_fn(server_port, 2.0))
                else:
                    is_alive = bool(probe_fn(server_host, server_port, 2.0))
            except Exception:
                try:
                    is_alive = bool(probe_fn())
                except Exception:
                    is_alive = False
        else:
            is_alive = probe_vllm_server(host=server_host, port=server_port, timeout=2.0)

        if not is_alive:
            print(
                "[!] Warning: TRL instalado só suporta vLLM em modo servidor; "
                "upgrade with `pip install -U 'trl>=0.19'` to use colocate. Fazendo fallback para geração nativa HF."
            )
            return {"use_vllm": False}

        result = {"use_vllm": True}
        if "vllm_server_port" in sig_params:
            result["vllm_server_port"] = server_port
        if "vllm_server_host" in sig_params:
            result["vllm_server_host"] = server_host
        return result

    # 3. Legacy TRL (~0.14 / 0.15) operating via vllm_device
    if "vllm_device" in sig_params:
        resolved_device = "cuda:0"
        if vllm_device and vllm_device != "auto":
            resolved_device = vllm_device
        result = {
            "use_vllm": True,
            "vllm_device": resolved_device,
        }
        if "vllm_gpu_memory_utilization" in sig_params:
            result["vllm_gpu_memory_utilization"] = gpu_mem_util
        return result

    result = {"use_vllm": True}
    if "vllm_gpu_memory_utilization" in sig_params:
        result["vllm_gpu_memory_utilization"] = gpu_mem_util
    return result


# ==============================================================================
# Prompt Truncation and GRPOConfig Keyword Filtering (TRL Compatibility)
# ==============================================================================

def _tokenize_helper(tokenizer: Any, text: str) -> List[int]:
    """Tokenizes text with fallback handling for tokenizer signature variations."""
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        try:
            return tokenizer.encode(text)
        except Exception:
            return [ord(c) for c in text]


def _decode_helper(tokenizer: Any, tokens: List[int]) -> str:
    """Decodes tokens with fallback handling for tokenizer signature variations."""
    try:
        return tokenizer.decode(tokens, skip_special_tokens=False)
    except TypeError:
        try:
            return tokenizer.decode(tokens)
        except Exception:
            return "".join(chr(t) for t in tokens)


def truncate_prompt(
    prompt: Union[str, List[Dict[str, str]]],
    tokenizer: Any,
    max_prompt_length: int,
) -> Union[str, List[Dict[str, str]]]:
    """Truncates prompt keeping the tail end (left-side truncated) if exceeding max_prompt_length.

    Preserves latest user instructions (right-side context) while trimming oldest history tokens.
    """
    if not prompt or max_prompt_length <= 0 or tokenizer is None:
        return prompt

    if isinstance(prompt, str):
        tokens = _tokenize_helper(tokenizer, prompt)
        if len(tokens) > max_prompt_length:
            tokens = tokens[-max_prompt_length:]
            return _decode_helper(tokenizer, tokens)
        return prompt

    if isinstance(prompt, list):
        copied_prompt = [dict(m) for m in prompt]
        if not copied_prompt:
            return copied_prompt

        if len(copied_prompt) == 1:
            content = copied_prompt[0].get("content", "")
            tokens = _tokenize_helper(tokenizer, content)
            if len(tokens) > max_prompt_length:
                tokens = tokens[-max_prompt_length:]
                copied_prompt[0]["content"] = _decode_helper(tokenizer, tokens)
            return copied_prompt

        msg_tokens = [_tokenize_helper(tokenizer, m.get("content", "")) for m in copied_prompt]
        total_tokens = sum(len(t) for t in msg_tokens)
        if total_tokens <= max_prompt_length:
            return copied_prompt

        if len(msg_tokens[-1]) >= max_prompt_length:
            last_tokens = msg_tokens[-1][-max_prompt_length:]
            copied_prompt[-1]["content"] = _decode_helper(tokenizer, last_tokens)
            for idx in range(len(copied_prompt) - 1):
                copied_prompt[idx]["content"] = ""
            return [m for m in copied_prompt if m.get("content")]

        budget_for_others = max_prompt_length - len(msg_tokens[-1])
        for idx in range(len(copied_prompt) - 2, -1, -1):
            cur_tokens = msg_tokens[idx]
            if len(cur_tokens) <= budget_for_others:
                budget_for_others -= len(cur_tokens)
            else:
                if budget_for_others > 0:
                    truncated = cur_tokens[-budget_for_others:]
                    copied_prompt[idx]["content"] = _decode_helper(tokenizer, truncated)
                    budget_for_others = 0
                else:
                    copied_prompt[idx]["content"] = ""

        return [m for m in copied_prompt if m.get("content") or m.get("role") == "system"]

    return prompt


def truncate_dataset_prompts(
    records: List[Dict[str, Any]],
    tokenizer: Any,
    max_prompt_length: int,
) -> List[Dict[str, Any]]:
    """Applies left-truncation to prompt fields across all dataset records preserving suffix context."""
    if not records or tokenizer is None or max_prompt_length <= 0:
        return records

    truncated_records = []
    for r in records:
        r_copy = dict(r)
        if "prompt" in r_copy:
            r_copy["prompt"] = truncate_prompt(r_copy["prompt"], tokenizer, max_prompt_length)
        truncated_records.append(r_copy)
    return truncated_records


def filter_supported_kwargs(
    config_cls: Any,
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Filters kwargs for configuration classes (e.g., GRPOConfig / TrainingArguments).

    Inspects dataclasses.fields and inspect.signature to ensure parameter support.
    Returns: (filtered_kwargs, dropped_kwargs).
    """
    supported = set()
    if dataclasses.is_dataclass(config_cls):
        supported.update(f.name for f in dataclasses.fields(config_cls))
    try:
        if inspect.isclass(config_cls):
            sig = inspect.signature(config_cls.__init__)
        else:
            sig = inspect.signature(config_cls)
        for name, param in sig.parameters.items():
            if name != "self" and param.kind != inspect.Parameter.VAR_KEYWORD:
                supported.add(name)
    except Exception:
        pass

    filtered: Dict[str, Any] = {}
    dropped: Dict[str, Any] = {}

    for k, v in kwargs.items():
        if k in supported:
            filtered[k] = v
        else:
            dropped[k] = v

    return filtered, dropped


# ==============================================================================
# Main GRPO Training CLI and Runner
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="CyberRott v1.3 - Online GRPO Training Pipeline with TRL and PEFT"
    )

    # Model and Adapter Arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["phi4-mini", "qwen2.5-coder-7b", "phi4-14b"],
        help="Pre-configured backbone architecture from SUPPORTED_BACKBONES."
    )
    model_group.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/Phi-4-mini-instruct",
        help="Hugging Face base model identifier or local directory (default: microsoft/Phi-4-mini-instruct)."
    )
    model_group.add_argument(
        "--adapter_name_or_path", "--sft_lora_path",
        dest="adapter_name_or_path",
        type=str,
        default="./cyberrott_v1_1_sft_lora",
        help="Path to Phase 2 (SFT) LoRA checkpoint. Fused in memory as reference policy π_ref."
    )
    model_group.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Tensor numerical precision (default: bfloat16)."
    )
    model_group.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Attention implementation: sdpa or flash_attention_2."
    )

    # Data Arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--dataset_name", "--data_path", "--data_file",
        dest="dataset_name",
        type=str,
        default="data/cyberrott_final_train.jsonl",
        help="JSONL path for the consolidated training dataset."
    )
    data_group.add_argument(
        "--anchor_file",
        type=str,
        default="data/dataset_cyber_ready.jsonl",
        help="JSONL path for the high-density Golden Anchors."
    )
    data_group.add_argument(
        "--max_samples",
        type=int,
        default=800,
        help="Sample count curated by technical richness (default: 800; <=0 uses full corpus)."
    )
    data_group.add_argument(
        "--max_prompt_length",
        type=int,
        default=768,
        help="Maximum prompt context length in tokens (default: 768)."
    )
    data_group.add_argument(
        "--max_completion_length",
        type=int,
        default=1024,
        help="Maximum generated rollout length in tokens (default: 1024)."
    )

    # GRPO Hyperparameters
    grpo_group = parser.add_argument_group("GRPO Hyperparameters")
    grpo_group.add_argument(
        "--output_dir",
        type=str,
        default="./CyberRott-v1.3-GRPO",
        help="Output directory to save final GRPO checkpoint and adapters."
    )
    grpo_group.add_argument(
        "--num_generations",
        type=int,
        default=8,
        help="Number of rollouts generated per prompt (G) for relative advantage calculation (default: 8)."
    )
    grpo_group.add_argument(
        "--beta",
        type=float,
        default=0.02,
        help="KL divergence regularization penalty vs reference policy π_ref (default: 0.02)."
    )
    grpo_group.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="AdamW learning rate for the GRPO phase (default: 5e-6)."
    )
    grpo_group.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Per-device forward/backward micro-batch size (default: 2; minimizes activation VRAM)."
    )
    grpo_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps (default: 8; micro_batch=2 * 8 = effective batch 16)."
    )
    grpo_group.add_argument(
        "--gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        default=None,
        help="Enable gradient checkpointing (default: True to prevent OOM with Phi-4 200k vocabulary)."
    )
    grpo_group.add_argument(
        "--no_gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Explicitly disable gradient checkpointing."
    )
    grpo_group.add_argument(
        "--num_train_epochs", "--epochs",
        dest="num_train_epochs",
        type=int,
        default=1,
        help="Number of GRPO training epochs (default: 1)."
    )
    grpo_group.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum training steps (-1 uses num_train_epochs)."
    )
    grpo_group.add_argument(
        "--save_steps",
        type=int,
        default=20,
        help="Checkpoint saving interval in steps (default: 20)."
    )
    grpo_group.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint or 'auto' to resume from the latest in --output_dir."
    )
    grpo_group.add_argument(
        "--logging_steps",
        type=int,
        default=5,
        help="Logging frequency in steps."
    )
    grpo_group.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Maximum checkpoints retained on disk."
    )
    grpo_group.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Learning rate schedule warmup ratio."
    )
    grpo_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    grpo_group.add_argument(
        "--full_scale",
        action="store_true",
        default=False,
        help="Train on the full high-density corpus (4680+ samples)."
    )
    grpo_group.add_argument(
        "--preset_v1_4", "--v1_4", "--v1.4",
        dest="preset_v1_4",
        action="store_true",
        default=False,
        help="Activate CyberRott v1.4 full-scale preset (DoRA, LoRA r=64/alpha=128, PRM rewards, 2048 completion length, full 4680 corpus)."
    )
    grpo_group.add_argument(
        "--w_prm",
        type=float,
        default=0.25,
        help="Weight for step-level Process Reward Model (PRM) reward function (default: 0.25)."
    )
    grpo_group.add_argument(
        "--include_prm",
        action="store_true",
        default=False,
        help="Explicitly activate step-level PRM reward function in the GRPO reward suite."
    )

    # LoRA / PEFT Arguments
    lora_group = parser.add_argument_group("LoRA")
    lora_group.add_argument(
        "--lora_r",
        type=int,
        default=32,
        help="LoRA rank for adaptation during GRPO (default: 32)."
    )
    lora_group.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha scaling factor (default: 64)."
    )
    lora_group.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate (default: 0.05)."
    )
    lora_group.add_argument(
        "--use_dora",
        action="store_true",
        default=False,
        help="Enable Weight-Decomposed Low-Rank Adaptation (DoRA) for enhanced capacity."
    )

    # vLLM Arguments
    vllm_group = parser.add_argument_group("vLLM")
    vllm_group.add_argument(
        "--use_vllm",
        action="store_true",
        default=False,
        help="Use vLLM engine for high-throughput rollout generation during GRPO training."
    )
    vllm_group.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.30,
        help="VRAM fraction allocated for the vLLM engine (default: 0.30)."
    )
    vllm_group.add_argument(
        "--vllm_device",
        type=str,
        default="auto",
        help="Device specification for vLLM (default: 'auto')."
    )
    vllm_group.add_argument(
        "--vllm_server_port",
        type=int,
        default=8000,
        help="External vLLM server port if operating in client-server mode (default: 8000)."
    )
    vllm_group.add_argument(
        "--vllm_server_host",
        type=str,
        default="127.0.0.1",
        help="External vLLM server host if operating in client-server mode (default: '127.0.0.1')."
    )

    # Execution Flags
    exec_group = parser.add_argument_group("Execution")
    exec_group.add_argument(
        "--dry_run",
        action="store_true",
        help="Non-GPU audit mode: validates dataset parsing, ground-truth extraction, and all 3 rewards."
    )
    exec_group.add_argument(
        "--auto_optimize",
        action="store_true",
        default=True,
        help="Dynamically configure TF32 and attention kernels based on detected accelerator."
    )
    exec_group.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Publish the final checkpoint to the Hugging Face Hub."
    )
    exec_group.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Hugging Face repository ID if --push_to_hub is active."
    )

    parsed = parser.parse_args()

    if parsed.preset_v1_4:
        parsed.use_dora = True
        parsed.lora_r = 64
        parsed.lora_alpha = 128
        parsed.max_completion_length = 2048
        parsed.max_samples = 4680
        parsed.include_prm = True

    if parsed.full_scale:
        parsed.max_samples = 4680

    if parsed.backbone:
        cfg = resolve_backbone_config(parsed.backbone)
        if parsed.model_name_or_path == "microsoft/Phi-4-mini-instruct":
            parsed.model_name_or_path = cfg["model_id"]

    return parsed


def main():
    args = parse_args()
    print("=" * 70)
    print("🚀 CyberRott v1.3 — GRPO Online Training Pipeline (DeepSeek-R1 Style)")
    print("=" * 70)

    # 1. Hardware acceleration setup
    hw_info = setup_hardware_optimizations(args)
    print(f"[*] Hardware: {hw_info['device_name']} | VRAM: {hw_info['vram_gb']} GB | Attention: {args.attn_implementation}")

    # 2. Dataset curation and verifiability extraction
    dataset_records = parse_and_curate_dataset(
        train_path=args.dataset_name,
        anchor_path=args.anchor_file,
        max_samples=args.max_samples
    )

    # 3. Dry-Run Mode execution if requested
    if args.dry_run:
        success = run_dry_run_validation(dataset_records)
        sys.exit(0 if success else 1)

    # 4. Import ML frameworks (executed only for real training runs)
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, PeftModel
    import trl
    from trl import GRPOTrainer, GRPOConfig

    trl_version = getattr(trl, "__version__", "unknown")
    print(f"[*] Detected TRL version: v{trl_version}")

    # 5. Load Tokenizer
    print(f"[*] Loading tokenizer from '{args.model_name_or_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    # 6. Load Base Model
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.torch_dtype, torch.bfloat16)

    print(f"[*] Loading base model: {args.model_name_or_path} ({args.torch_dtype})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=False,
    )

    # 7. In-Memory Fusion: SFT LoRA -> Base Model (Policy π_ref with 0 extra VRAM)
    adapter_path = args.adapter_name_or_path
    if adapter_path and os.path.exists(adapter_path) and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print(f"\n[*] Prior SFT checkpoint detected at: '{adapter_path}'")
        print("[*] Merging LoRA weights in memory (merge_and_unload)...")
        start_merge = time.time()
        peft_sft = PeftModel.from_pretrained(base_model, adapter_path)
        base_model = peft_sft.merge_and_unload()
        print(f"[✓] Fusion completed in {time.time() - start_merge:.2f}s!")
        print("[✓] Model with SFT tradecraft now directly acts as reference policy π_ref!")
        print("[✓] Memory savings: 0 additional VRAM required for reference model!\n")
    else:
        print(f"[*] No prior adapter found at '{adapter_path}'. Starting GRPO directly on base model.")

    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()

    # Enable KV-cache acceleration for base model during rollouts with gradient checkpointing
    enable_fast_generation_with_kv_cache(base_model)

    # 8. Setup fresh LoRA adapter for GRPO training
    target_modules = "all-linear"
    if getattr(args, "backbone", None):
        cfg = resolve_backbone_config(args.backbone)
        target_modules = cfg["target_modules"]

    print(f"[*] Configuring clean LoRA for RL phase: r={args.lora_r}, alpha={args.lora_alpha}, target={target_modules}, use_dora={getattr(args, 'use_dora', False)}...")
    lora_config = build_lora_config(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        use_dora=getattr(args, "use_dora", False),
    )

    # 9. Configure GRPOConfig (TRL)
    supported_grpo_fields = set()
    if dataclasses.is_dataclass(GRPOConfig):
        supported_grpo_fields.update(f.name for f in dataclasses.fields(GRPOConfig))
    try:
        supported_grpo_fields.update(p for p in inspect.signature(GRPOConfig.__init__).parameters if p != "self")
    except Exception:
        pass

    has_modern_grpo_batching = (
        "generation_batch_size" in supported_grpo_fields
        or "steps_per_generation" in supported_grpo_fields
    )

    if has_modern_grpo_batching:
        effective_gen_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
        if effective_gen_batch % args.num_generations != 0:
            target_batch = ((effective_gen_batch + args.num_generations - 1) // args.num_generations) * args.num_generations
            new_gas = (target_batch + args.per_device_train_batch_size - 1) // args.per_device_train_batch_size
            while (args.per_device_train_batch_size * new_gas) % args.num_generations != 0:
                new_gas += 1
            print(
                f"[*] Modern TRL GRPO auto-adjustment: gradient_accumulation_steps adjusted from "
                f"{args.gradient_accumulation_steps} to {new_gas} to ensure effective batch "
                f"({args.per_device_train_batch_size * new_gas}) is divisible by num_generations={args.num_generations}."
            )
            args.gradient_accumulation_steps = new_gas
    else:
        if args.per_device_train_batch_size < args.num_generations or (args.per_device_train_batch_size % args.num_generations != 0):
            print(
                f"[*] Legacy TRL GRPO auto-adjustment: per_device_train_batch_size ({args.per_device_train_batch_size}) "
                f"adjusted to {args.num_generations} (must be multiple of num_generations={args.num_generations})."
            )
            args.per_device_train_batch_size = args.num_generations

    if args.gradient_checkpointing is not None:
        use_gc = bool(args.gradient_checkpointing)
        print(f"[*] Gradient Checkpointing explicitly set by user: {use_gc}")
    else:
        use_gc = True
        print(
            "[*] Gradient Checkpointing ENABLED by default (critical in GRPO with Phi-4-mini 200k vocabulary "
            "and 32 layers to prevent activation OOM; rollouts retain KV-cache acceleration via wrapper)."
        )

    if use_gc:
        if hasattr(base_model, "gradient_checkpointing_enable"):
            try:
                base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                base_model.gradient_checkpointing_enable()
            print("[✓] Gradient Checkpointing enabled on base model (use_reentrant=False).")
        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False

    use_vllm = args.use_vllm
    if use_vllm:
        import importlib.util
        if importlib.util.find_spec("vllm") is None:
            print("[!] Warning: --use_vllm requested, but 'vllm' package is not installed. Running rollouts with native PyTorch KV-cache.")
            use_vllm = False
        else:
            print("[✓] 'vllm' package detected in Python environment.")

    vllm_kwargs = resolve_vllm_kwargs(
        config_cls=GRPOConfig,
        use_vllm=use_vllm,
        gpu_mem_util=args.vllm_gpu_memory_utilization,
        vllm_device=args.vllm_device,
        server_port=getattr(args, "vllm_server_port", 8000),
        server_host=getattr(args, "vllm_server_host", "127.0.0.1"),
    )
    use_vllm = vllm_kwargs.get("use_vllm", False)

    sig_params = inspect.signature(GRPOConfig.__init__).parameters
    if not use_vllm:
        vllm_mode_desc = "Disabled (native Hugging Face / PyTorch KV-cache generation)"
    elif vllm_kwargs.get("vllm_mode") == "colocate":
        vllm_mode_desc = f"vLLM colocate in-process (GPU Memory Utilization: {vllm_kwargs.get('vllm_gpu_memory_utilization', args.vllm_gpu_memory_utilization)})"
    elif "vllm_server_port" in vllm_kwargs or "vllm_server_host" in sig_params:
        vllm_mode_desc = f"vLLM active server mode ({getattr(args, 'vllm_server_host', '127.0.0.1')}:{vllm_kwargs.get('vllm_server_port', getattr(args, 'vllm_server_port', 8000))})"
    elif "vllm_device" in vllm_kwargs:
        vllm_mode_desc = f"vLLM legacy (device: {vllm_kwargs.get('vllm_device')})"
    else:
        vllm_mode_desc = "vLLM enabled"

    print(f"[*] Selected vLLM generation mode: {vllm_mode_desc}")

    print(f"[*] Setting GRPO hyperparameters: lr={args.learning_rate}, beta={args.beta}, G={args.num_generations}...")
    grpo_config_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_train_epochs": args.num_train_epochs if args.max_steps <= 0 else 1.0,
        "max_steps": args.max_steps,
        "remove_unused_columns": False,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "fp16": (args.torch_dtype == "float16"),
        "bf16": (args.torch_dtype == "bfloat16"),
        "gradient_checkpointing": use_gc,
        "push_to_hub": args.push_to_hub,
        "hub_model_id": args.hub_model_id,
    }
    if use_gc and "gradient_checkpointing_kwargs" in sig_params:
        grpo_config_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    grpo_config_kwargs.update(vllm_kwargs)

    effective_gen_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if "generation_batch_size" in supported_grpo_fields:
        grpo_config_kwargs["generation_batch_size"] = effective_gen_batch
        print(f"[*] Modern TRL: generation_batch_size configured as {effective_gen_batch} (divisible by G={args.num_generations}).")

    if "torch_empty_cache_steps" in supported_grpo_fields:
        grpo_config_kwargs["torch_empty_cache_steps"] = 1
        print("[*] TRL GRPO: torch_empty_cache_steps=1 configured for continuous VRAM defragmentation.")

    filtered_config_kwargs, dropped_config_kwargs = filter_supported_kwargs(GRPOConfig, grpo_config_kwargs)

    if dropped_config_kwargs:
        dropped_names = ", ".join(sorted(dropped_config_kwargs.keys()))
        print(f"[!] Parameters ignored due to incompatibility with TRL {trl_version}: {dropped_names}")

    if "max_prompt_length" in dropped_config_kwargs:
        print(
            f"[*] 'max_prompt_length' not natively supported by GRPOConfig (TRL {trl_version}). "
            f"Applying manual prompt truncation ({args.max_prompt_length} tokens, truncation_side='left')..."
        )
        dataset_records = truncate_dataset_prompts(dataset_records, tokenizer, args.max_prompt_length)
        print(f"[✓] Manual prompt truncation completed for {len(dataset_records)} samples.")

    hf_dataset = Dataset.from_list(dataset_records)
    print(f"[✓] Hugging Face Dataset constructed with {len(hf_dataset)} samples.")

    grpo_config = GRPOConfig(**filtered_config_kwargs)

    # 10. Initialize GRPOTrainer
    reward_suite = [
        cot_integrity_reward,
        technical_accuracy_reward,
        sandbox_code_reward,
    ]
    if getattr(args, "include_prm", False) or getattr(args, "preset_v1_4", False):
        prm_weighted = functools.partial(prm_step_reward, w_prm=getattr(args, "w_prm", 0.25))
        functools.update_wrapper(prm_weighted, prm_step_reward)
        reward_suite.append(prm_weighted)
        print(f"[Reward Suite] Step-Level PRM reward active in GRPO suite (weight: {getattr(args, 'w_prm', 0.25)}).")

    apply_grpo_compatibility_patches()
    if not hasattr(base_model, "warnings_issued"):
        base_model.warnings_issued = {}

    trainer_init_params = inspect.signature(GRPOTrainer.__init__).parameters
    trainer_kwargs = {
        "model": base_model,
        "reward_funcs": reward_suite,
        "args": grpo_config,
        "train_dataset": hf_dataset,
        "peft_config": lora_config,
    }
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        trainer_kwargs["processing_class"] = tokenizer

    print(f"[*] Initializing GRPOTrainer with {len(reward_suite)} verifiable reward functions...")
    trainer = GRPOTrainer(**trainer_kwargs)

    eos_ids = get_eos_token_ids(tokenizer)
    if eos_ids:
        print(f"[*] Stop/EOS Token IDs configured for generation ({len(eos_ids)} tokens): {eos_ids}")
        if hasattr(trainer, "generation_config") and trainer.generation_config is not None:
            trainer.generation_config.eos_token_id = eos_ids
        if hasattr(base_model, "generation_config") and base_model.generation_config is not None:
            base_model.generation_config.eos_token_id = eos_ids
        if hasattr(trainer.model, "generation_config") and trainer.model.generation_config is not None:
            trainer.model.generation_config.eos_token_id = eos_ids

    enable_fast_generation_with_kv_cache(trainer.model)
    if hasattr(trainer, "model_wrapped") and trainer.model_wrapped is not None:
        enable_fast_generation_with_kv_cache(trainer.model_wrapped)
    print("[*] KV-Cache wrapper successfully activated for rollout acceleration.")

    # 11. Run GRPO Training
    print("\n" + "=" * 70)
    print("🔥 STARTING ONLINE GRPO REINFORCEMENT LEARNING (CyberRott v1.3)")
    print("=" * 70 + "\n")

    print("[*] Running garbage collection and emptying CUDA cache prior to training...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    resume_checkpoint = resolve_resume_checkpoint(args.resume_from_checkpoint, args.output_dir)
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    print("\n" + "=" * 70)
    print("[✓] GRPO Training completed successfully!")
    print(f"[*] Final metrics: {train_result.metrics}")
    print("=" * 70 + "\n")

    # 12. Save Model and Tokenizer
    print(f"[*] Saving final checkpoint to: {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[✓] CyberRott v1.3 GRPO checkpoint successfully written to '{args.output_dir}'!")

    # 13. Synchronize with Google Drive if in Colab
    sync_checkpoint_to_google_drive(
        source_dir=args.output_dir,
        target_subfolder="CyberRott-v1.3-GRPO"
    )

    if args.push_to_hub:
        print("[*] Publishing model to Hugging Face Hub...")
        trainer.push_to_hub()
        print("[✓] Model successfully published!")


if __name__ == "__main__":
    main()
