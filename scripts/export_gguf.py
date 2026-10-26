#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CyberRott 1.3 - GGUF Export and Quantization Pipeline
=====================================================
Automates the full post-training workflow for local Ollama / llama.cpp deployment:
1. Sequential Adapter Fusion (Base Model + SFT LoRA + GRPO RL -> Standalone).
2. Download and compilation/configuration of llama.cpp.
3. Conversion of Hugging Face weights to base GGUF (F16).
4. High-performance quantization (default: Q4_K_M for ~2.3 GB, or Q8_0).
5. Automated generation of ready-to-run Modelfile for immediate Ollama execution.

Usage Example:
    python scripts/export_gguf.py \
        --sft_adapter /content/drive/MyDrive/CyberRott_Checkpoints/cyberrott_v1_1_sft_lora \
        --grpo_adapter /content/drive/MyDrive/CyberRott_Checkpoints/CyberRott_v1_3_GRPO \
        --quant_type Q4_K_M \
        --output_dir ./exports
"""

import os
import sys

# Disable broken TensorFlow in Colab Python 3.13 environments
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
sys.modules["tensorflow"] = None

import shutil
import argparse
import subprocess
from pathlib import Path

# Disable torchao availability checks for Google Colab compatibility
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
    import peft.tuners.lora.torchao
    peft.tuners.lora.torchao.is_torchao_available = lambda: False
except Exception:
    pass


def run_cmd(command: str, description: str = ""):
    """Executes a shell command with error handling and structured console logging."""
    if description:
        print(f"\n[*] {description}...")
    print(f"    $ {command}")
    res = subprocess.run(command, shell=True, text=True)
    if res.returncode != 0:
        print(f"[!] Execution failed for: {command}")
        sys.exit(res.returncode)


def export_pipeline(
    base_model: str,
    sft_adapter: str,
    grpo_adapter: str,
    quant_type: str,
    output_dir: str,
    torch_dtype: str = "bfloat16",
    ollama_model_name: str = "cyberrott-v1.3",
    skip_merge: bool = False,
    standalone_path: str = None,
):
    is_f16 = quant_type.lower() in ["f16", "fp16", "none"]
    mode_desc = "Pure Unquantized F16 (Full 16-bit Precision)" if is_f16 else f"{quant_type} Quantization"

    print("=" * 70)
    print(f"🛡️  CyberRott v1.3 — GGUF Export Pipeline ({mode_desc})")
    print("=" * 70)
    print(f"Base Model:       {base_model}")
    print(f"SFT Adapter:      {sft_adapter or 'None'}")
    print(f"GRPO Adapter:     {grpo_adapter}")
    print(f"Target Format:    {mode_desc}")
    print(f"Output Directory: {output_dir}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    standalone_dir = standalone_path or os.path.join(output_dir, "CyberRott-v1.3-Standalone")
    f16_gguf_path = os.path.join(output_dir, "cyberrott-v1.3-f16.gguf")
    quant_gguf_path = os.path.join(output_dir, f"cyberrott-v1.3-{quant_type.lower()}.gguf")
    modelfile_path = os.path.join(output_dir, "Modelfile")
    llama_cpp_dir = os.path.join(output_dir, "llama.cpp")

    # -------------------------------------------------------------------------
    # Step 1: Weight Fusion (Merge & Unload)
    # -------------------------------------------------------------------------
    if not skip_merge:
        print("\n" + "=" * 70)
        print("STAGE 1: SEQUENTIAL ADAPTER FUSION (MERGE & UNLOAD)")
        print("=" * 70)
        
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from scripts.merge_lora import merge_and_save

        merge_and_save(
            base_model_path=base_model,
            adapter_path=grpo_adapter,
            output_path=standalone_dir,
            torch_dtype=torch_dtype,
            sft_adapter_path=sft_adapter,
        )
    else:
        print(f"\n[*] Skipping fusion stage. Using existing standalone model at: {standalone_dir}")

    # -------------------------------------------------------------------------
    # Step 2: Clone and Configure llama.cpp
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 2: LLAMA.CPP ENGINE SETUP")
    print("=" * 70)

    if not os.path.exists(llama_cpp_dir):
        run_cmd(
            f"git clone --depth 1 https://github.com/ggerganov/llama.cpp.git {llama_cpp_dir}",
            "Cloning llama.cpp repository"
        )
    else:
        print(f"[✓] llama.cpp repository already present at: {llama_cpp_dir}")

    # Install conversion dependencies for llama.cpp
    run_cmd(
        f"{sys.executable} -m pip install -q -r {llama_cpp_dir}/requirements.txt",
        "Installing llama.cpp conversion dependencies"
    )

    # -------------------------------------------------------------------------
    # Step 3: HuggingFace -> GGUF F16 Conversion
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 3: WEIGHT CONVERSION TO GGUF FORMAT (F16)")
    print("=" * 70)

    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        # Fallback for recent llama.cpp repository layouts
        convert_script = os.path.join(llama_cpp_dir, "convert-hf-to-gguf.py")

    run_cmd(
        f"{sys.executable} {convert_script} {standalone_dir} --outtype f16 --outfile {f16_gguf_path}",
        "Converting Hugging Face tensors to GGUF F16 file"
    )

    # -------------------------------------------------------------------------
    # Step 4: High-Efficiency Quantization (Optional if quant_type != f16)
    # -------------------------------------------------------------------------
    is_f16 = quant_type.lower() in ["f16", "fp16", "none"]
    if is_f16:
        print("\n" + "=" * 70)
        print("STAGE 4: FULL F16 PRECISION SELECTED (QUANTIZATION SKIPPED)")
        print("=" * 70)
        print(f"[✓] GGUF model generated in full unquantized float16 precision: {f16_gguf_path}")
        quant_gguf_path = f16_gguf_path
    else:
        print("\n" + "=" * 70)
        print(f"STAGE 4: COMPILATION AND QUANTIZATION TO {quant_type}")
        print("=" * 70)

        quantize_bin = os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize")
        if not os.path.exists(quantize_bin):
            quantize_bin_alt = os.path.join(llama_cpp_dir, "llama-quantize")
            if os.path.exists(quantize_bin_alt):
                quantize_bin = quantize_bin_alt
            else:
                build_dir = os.path.join(llama_cpp_dir, "build")
                run_cmd(f"cmake -B {build_dir} {llama_cpp_dir}", "Configuring CMake for build")
                run_cmd(f"cmake --build {build_dir} --config Release -j --target llama-quantize", "Compiling llama-quantize binary")

        run_cmd(
            f"{quantize_bin} {f16_gguf_path} {quant_gguf_path} {quant_type}",
            f"Executing {quant_type} quantization on model"
        )

    # -------------------------------------------------------------------------
    # Step 5: Automatic Modelfile Generation for Ollama
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 5: GENERATING MODELFILE FOR OLLAMA")
    print("=" * 70)

    modelfile_content = f"""FROM {os.path.abspath(quant_gguf_path)}

TEMPLATE \"\"\"<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"

SYSTEM \"\"\"You are CyberRott, a specialized reasoning AI for offensive, defensive, and vulnerability analysis with deep 7-stage analytical reasoning.\"\"\"

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|end|>"
PARAMETER stop "<|endoftext|>"
"""

    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    final_size_mb = os.path.getsize(quant_gguf_path) / (1024 * 1024)
    print(f"[✓] Modelfile successfully generated at: {modelfile_path}")
    print(f"[✓] Final quantized binary:              {quant_gguf_path} ({final_size_mb:.1f} MB / {final_size_mb/1024:.2f} GB)")

    print("\n" + "=" * 70)
    print("🎉 EXPORT COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("How to run locally in Ollama:")
    print(f"  1. Change directory: cd {os.path.abspath(output_dir)}")
    print(f"  2. Register model:   ollama create {ollama_model_name} -f Modelfile")
    print(f"  3. Launch model:     ollama run {ollama_model_name}")
    print("=" * 70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated LoRA Merge + GGUF Conversion + Quantization Pipeline for CyberRott 1.3"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/Phi-4-mini-instruct",
        help="Hugging Face base model identifier or local path (default: microsoft/Phi-4-mini-instruct)."
    )
    parser.add_argument(
        "--sft_adapter",
        type=str,
        default="./cyberrott_v1_1_sft_lora",
        help="Path to the supervised fine-tuning (SFT) adapter directory (optional, can be None)."
    )
    parser.add_argument(
        "--grpo_adapter", "--adapter",
        dest="grpo_adapter",
        type=str,
        default="./CyberRott-v1.2-GRPO",
        help="Path to the LoRA adapter trained with GRPO in Phase 3."
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="f16",
        choices=["f16", "F16", "fp16", "FP16", "Q4_K_M", "Q4_0", "Q5_K_M", "Q8_0"],
        help="Quantization method for Ollama/llama.cpp deployment (default: f16 unquantized)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./CyberRott-v1.3-Export",
        help="Output directory where standalone model, GGUFs, and Modelfile will be stored."
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Numerical precision for weight merging (default: bfloat16)."
    )
    parser.add_argument(
        "--ollama_model_name",
        type=str,
        default="cyberrott-v1.3",
        help="Model name to register for the 'ollama run' command (default: cyberrott-v1.3)."
    )
    parser.add_argument(
        "--skip_merge",
        action="store_true",
        help="Skip adapter merging if fused standalone weights already exist on disk."
    )
    parser.add_argument(
        "--standalone_path",
        type=str,
        default=None,
        help="Custom path to an existing standalone merged model (used with --skip_merge)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Normalize sft_adapter path if non-existent
    sft_path = args.sft_adapter
    if sft_path and not os.path.exists(sft_path):
        drive_candidate = "/content/drive/MyDrive/CyberRott_Checkpoints/cyberrott_v1_1_sft_lora"
        if os.path.exists(drive_candidate):
            sft_path = drive_candidate
        else:
            sft_path = None

    grpo_path = args.grpo_adapter
    if not os.path.exists(grpo_path):
        for candidate in [
            "/content/drive/MyDrive/CyberRott_Checkpoints/CyberRott_v1_3_GRPO",
            "/content/drive/MyDrive/CyberRott_Checkpoints/CyberRott-v1.3-GRPO",
            "/content/drive/MyDrive/CyberRott_Checkpoints/CyberRott_v1_2_GRPO",
        ]:
            if os.path.exists(candidate):
                grpo_path = candidate
                break

    export_pipeline(
        base_model=args.base_model,
        sft_adapter=sft_path,
        grpo_adapter=grpo_path,
        quant_type=args.quant_type,
        output_dir=args.output_dir,
        torch_dtype=args.torch_dtype,
        ollama_model_name=args.ollama_model_name,
        skip_merge=args.skip_merge,
        standalone_path=args.standalone_path,
    )
