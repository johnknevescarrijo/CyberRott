#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para upload automático do CyberRott 1.0 para o Hugging Face Hub.
Cria o repositório se não existir, copia o MODEL_CARD.md como README.md oficial
e envia os pesos safetensors e o arquivo GGUF com suporte a resumo de upload.
"""

import os
import shutil
import argparse
from huggingface_hub import HfApi, create_repo

def upload_model(
    repo_name: str,
    standalone_folder: str = "./CyberRott-Phi4-mini-Standalone",
    gguf_file: str = "./CyberRott-Phi4-mini.gguf",
    model_card: str = "./MODEL_CARD.md",
    token: str = None
):
    api = HfApi(token=token)
    
    # 1. Identificar usuário
    user_info = api.whoami()
    username = user_info["name"]
    full_repo_id = f"{username}/{repo_name}"
    
    print("=" * 60)
    print("🚀 CyberRott 1.0 - Upload para o Hugging Face Hub")
    print(f"Usuário autenticado: {username}")
    print(f"Repositório Alvo:    https://huggingface.co/{full_repo_id}")
    print("=" * 60)

    # 2. Criar repositório no Hub (se não existir)
    print("1. Verificando/Criando repositório no Hugging Face...")
    create_repo(repo_id=full_repo_id, repo_type="model", exist_ok=True, token=token)
    print("   [✓] Repositório pronto!")

    # 3. Preparar MODEL_CARD.md como README.md no diretório standalone
    if os.path.exists(model_card):
        dest_readme = os.path.join(standalone_folder, "README.md")
        shutil.copy(model_card, dest_readme)
        print(f"   [✓] MODEL_CARD.md copiado como README.md oficial em {dest_readme}")

    # 4. Incluir GGUF no diretório para quem usa Ollama/llama.cpp
    if gguf_file and os.path.exists(gguf_file):
        dest_gguf = os.path.join(standalone_folder, os.path.basename(gguf_file))
        if not os.path.exists(dest_gguf):
            print(f"2. Vinculando arquivo GGUF ({gguf_file}) ao repositório...")
            try:
                os.link(gguf_file, dest_gguf) # Hard link economiza espaço em disco
            except Exception:
                shutil.copy(gguf_file, dest_gguf)
        print("   [✓] Arquivo GGUF pronto para upload.")

    # 5. Fazer upload da pasta inteira com progresso e retry automático
    print("\n3. Iniciando upload dos pesos (safetensors + GGUF + configs)...")
    print("   Isso pode levar alguns minutos dependendo da sua velocidade de upload.")
    
    api.upload_folder(
        folder_path=standalone_folder,
        repo_id=full_repo_id,
        repo_type="model"
    )

    print("\n" + "=" * 60)
    print("🎉 SUCESSO! O CyberRott 1.0 está no ar para o mundo todo!")
    print(f"Acesse em: https://huggingface.co/{full_repo_id}")
    print(f"Para rodar direto no Ollama via Hugging Face:")
    print(f"   ollama run hf.co/{full_repo_id}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload do CyberRott para o Hugging Face Hub")
    parser.add_argument("--repo_name", default="CyberRott-Phi4-mini-Instruct", help="Nome do repositório no Hugging Face")
    parser.add_argument("--folder", default="./CyberRott-Phi4-mini-Standalone", help="Pasta com os pesos standalone")
    parser.add_argument("--gguf", default="./CyberRott-Phi4-mini.gguf", help="Arquivo GGUF para incluir no Hub")
    parser.add_argument("--token", default=None, help="Token HF com permissão Write (ou use huggingface-cli login)")
    args = parser.parse_args()

    upload_model(
        repo_name=args.repo_name,
        standalone_folder=args.folder,
        gguf_file=args.gguf,
        token=args.token
    )
