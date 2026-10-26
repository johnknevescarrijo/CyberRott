#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CyberRott (v1.3 GRPO) Hugging Face Hub Publication Pipeline.
============================================================
Creates the repository if it does not exist, copies MODEL_CARD.md as the official
README.md, and uploads safetensors weights and/or GGUF files with resumable upload support.
Supports the --readme-only flag to quickly synchronize Model Card metadata.
Supports automatic token resolution via CLI, HF_TOKEN env, Colab userdata secrets,
and huggingface_hub local cache.
"""

import os
import sys
import shutil
import argparse
from typing import Optional, List
from huggingface_hub import HfApi, create_repo, get_token

DEFAULT_REPO_NAME = "CyberRott-v1.3-GRPO"

CANDIDATE_FOLDERS: List[str] = [
    "/content/CyberRott-v1.3-BF16-Standalone",
    "./CyberRott-v1.3-Standalone",
    "./CyberRott-v1.3-BF16-Standalone",
    "/content/CyberRott-v1.2-BF16-Standalone",
    "./CyberRott-v1.2-BF16-Standalone",
    "./CyberRott-Phi4-mini-Standalone"
]

CANDIDATE_GGUFS: List[str] = [
    "./CyberRott-v1.3-GRPO.gguf",
    "/content/CyberRott-v1.3-GRPO.gguf",
    "./CyberRott-v1.2-mini.gguf",
    "/content/CyberRott-v1.2-mini.gguf",
    "./CyberRott-Phi4-mini.gguf"
]

CANDIDATE_MODELFILES: List[str] = [
    "./Modelfile_v1.3",
    "./Modelfile_v1.2",
    "./Modelfile"
]


def resolve_hf_token(cli_token: Optional[str] = None) -> Optional[str]:
    """
    Resolves Hugging Face access token with hierarchical precedence:
    1. Explicit CLI argument (--token)
    2. Environment variable (HF_TOKEN or HUGGING_FACE_HUB_TOKEN)
    3. Google Colab Secrets (userdata.get('HF_TOKEN'))
    4. Local huggingface_hub cache (prior login via huggingface-cli)
    """
    if cli_token and cli_token.strip():
        return cli_token.strip()

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()

    try:
        from google.colab import userdata  # type: ignore
        colab_token = userdata.get("HF_TOKEN")
        if colab_token and colab_token.strip():
            print("[✓] HF_TOKEN retrieved from Google Colab Secrets.")
            return colab_token.strip()
    except Exception:
        pass

    cached_token = get_token()
    if cached_token and cached_token.strip():
        return cached_token.strip()

    return None


def resolve_model_folder(specified_folder: Optional[str] = None) -> str:
    """
    Returns the first existing standalone weight directory among candidates,
    or the user-specified directory.
    """
    if specified_folder:
        if os.path.exists(specified_folder):
            return specified_folder
        print(f"[!] Specified folder '{specified_folder}' does not currently exist.")
        return specified_folder

    for candidate in CANDIDATE_FOLDERS:
        if os.path.exists(candidate) and os.path.isdir(candidate):
            # Check for safetensors or bin files
            files = os.listdir(candidate)
            if any(f.endswith(".safetensors") or f.endswith(".bin") or f == "config.json" for f in files):
                print(f"[✓] Weight directory dynamically located: {candidate}")
                return candidate

    # Default fallback
    fallback = "./CyberRott-v1.3-Standalone"
    print(f"[*] No pre-existing weight directory found. Using default fallback: {fallback}")
    return fallback


def resolve_gguf_file(specified_gguf: Optional[str] = None) -> Optional[str]:
    """Locates the GGUF file if available."""
    if specified_gguf and os.path.exists(specified_gguf):
        return specified_gguf

    for candidate in CANDIDATE_GGUFS:
        if os.path.exists(candidate):
            print(f"[✓] GGUF file dynamically located: {candidate}")
            return candidate

    return None


def upload_model(
    repo_name: str = DEFAULT_REPO_NAME,
    standalone_folder: Optional[str] = None,
    gguf_file: Optional[str] = None,
    model_card: str = "./MODEL_CARD.md",
    token: Optional[str] = None,
    readme_only: bool = False
):
    resolved_token = resolve_hf_token(token)
    if not resolved_token:
        print("\n❌ ERROR: Hugging Face token not found!")
        print("Please provide a token through one of the following methods:")
        print("  1. Terminal CLI argument: python scripts/upload_to_hf.py --token hf_xxx")
        print("  2. Environment variable: export HF_TOKEN=hf_xxx")
        print("  3. Google Colab: Add 'HF_TOKEN' to Secrets (key icon in left sidebar)")
        print("  4. Local login via CLI: huggingface-cli login")
        sys.exit(1)

    api = HfApi(token=resolved_token)
    
    # 1. Identify authenticated user
    try:
        user_info = api.whoami()
        username = user_info["name"]
    except Exception as e:
        print(f"❌ Failed to validate Hugging Face token: {e}")
        sys.exit(1)

    full_repo_id = f"{username}/{repo_name}"
    
    print("=" * 65)
    print("🛡️  CyberRott v1.3 GRPO — Publication to Hugging Face Hub")
    print(f"Authenticated User: {username}")
    print(f"Target Repository:  https://huggingface.co/{full_repo_id}")
    if readme_only:
        print("Operating Mode:     ⚡ Fast Model Card sync (--readme-only)")
    else:
        print("Operating Mode:     📦 Full upload (Weights + Model Card + Artifacts)")
    print("=" * 65)

    # 2. Create repository on Hub (if non-existent)
    print("1. Checking/Creating repository on Hugging Face...")
    create_repo(repo_id=full_repo_id, repo_type="model", exist_ok=True, token=resolved_token)
    print("   [✓] Repository confirmed and accessible!")

    # 3. Prepare upload directory and copy MODEL_CARD.md as official README.md
    target_folder = resolve_model_folder(standalone_folder)
    os.makedirs(target_folder, exist_ok=True)
    dest_readme = os.path.join(target_folder, "README.md")
    
    if os.path.exists(model_card):
        shutil.copy(model_card, dest_readme)
        print(f"   [✓] MODEL_CARD.md copied as official README.md to: {dest_readme}")
    else:
        print(f"   [!] Warning: {model_card} not found. Searching for local README.md...")
        dest_readme = os.path.join(target_folder, "README.md")

    # Copy Modelfile if available
    for mf_candidate in CANDIDATE_MODELFILES:
        if os.path.exists(mf_candidate):
            shutil.copy(mf_candidate, os.path.join(target_folder, "Modelfile"))
            print(f"   [✓] {mf_candidate} copied to repository as Modelfile.")
            break

    # If --readme-only, upload file directly and exit
    if readme_only:
        print("\n2. Uploading updated README.md / Model Card only...")
        source_file = dest_readme if os.path.exists(dest_readme) else model_card
        api.upload_file(
            path_or_fileobj=source_file,
            path_in_repo="README.md",
            repo_id=full_repo_id,
            repo_type="model"
        )
        print("   [✓] README.md successfully updated on Hugging Face!")
        print("\n" + "=" * 65)
        print("🎉 Model Card successfully updated on Hugging Face Hub!")
        print(f"Access at: https://huggingface.co/{full_repo_id}")
        print("=" * 65)
        return

    # 4. Include GGUF file in repository if available
    resolved_gguf = resolve_gguf_file(gguf_file)
    if resolved_gguf and os.path.exists(resolved_gguf):
        dest_gguf = os.path.join(target_folder, os.path.basename(resolved_gguf))
        if not os.path.exists(dest_gguf) and os.path.abspath(resolved_gguf) != os.path.abspath(dest_gguf):
            print(f"2. Linking GGUF file ({resolved_gguf}) to repository directory...")
            try:
                os.link(resolved_gguf, dest_gguf)  # Hard link preserves disk space
            except Exception:
                shutil.copy(resolved_gguf, dest_gguf)
        print(f"   [✓] GGUF file prepared for upload ({os.path.basename(dest_gguf)}).")
    else:
        print("2. [INFO] No local GGUF binary detected. Proceeding with BF16 / Safetensors weights.")

    # Integrity check: verify actual model weights exist before upload
    weight_files = [f for f in os.listdir(target_folder) if f.endswith((".safetensors", ".bin", ".gguf", ".pt"))]
    if not weight_files:
        print(f"\n❌ ERROR: No weight files (*.safetensors, *.gguf, *.bin) found in '{target_folder}'!")
        print("The folder contains only metadata/README. Ensure the weight fusion/export step succeeded before uploading.")
        sys.exit(1)

    # 5. Upload entire directory with progress tracking and automatic retries
    print(f"\n3. Starting upload of weight directory ({target_folder})...")
    print(f"   Detected tensor files ({len(weight_files)}): {weight_files}")
    print("   This may take several minutes depending on model size and bandwidth.")
    
    api.upload_folder(
        folder_path=target_folder,
        repo_id=full_repo_id,
        repo_type="model"
    )

    print("\n" + "=" * 65)
    print("🎉 SUCCESS! CyberRott v1.3 GRPO is published on the Hugging Face Hub!")
    print(f"Repository URL: https://huggingface.co/{full_repo_id}")
    print("Recommended inference commands:")
    print(f"   • Transformers / vLLM: vllm serve {full_repo_id}")
    print(f"   • Ollama Hub:          ollama run hf.co/{full_repo_id}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload CyberRott v1.3 to the Hugging Face Hub")
    parser.add_argument("--repo_name", default=DEFAULT_REPO_NAME, help="Hugging Face repository name (default: CyberRott-v1.3-GRPO)")
    parser.add_argument("--folder", default=None, help="Directory containing standalone weights (auto-resolved if omitted)")
    parser.add_argument("--gguf", default=None, help="GGUF binary file to include in the repository (optional)")
    parser.add_argument("--model_card", default="./MODEL_CARD.md", help="Model Card file to use as the official README.md")
    parser.add_argument("--token", default=None, help="Hugging Face token with Write access (or set HF_TOKEN env variable)")
    parser.add_argument("--readme-only", action="store_true", help="Synchronize only README.md / Model Card without uploading weights")
    args = parser.parse_args()

    upload_model(
        repo_name=args.repo_name,
        standalone_folder=args.folder,
        gguf_file=args.gguf,
        model_card=args.model_card,
        token=args.token,
        readme_only=args.readme_only
    )
