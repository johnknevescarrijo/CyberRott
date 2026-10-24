#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para upload automático do CyberRott 1.0 para o Hugging Face Hub.
Cria o repositório se não existir, copia o MODEL_CARD.md como README.md oficial
e envia os pesos safetensors e o arquivo GGUF com suporte a resumo de upload.
Suporta a flag --readme-only para sincronizar rapidamente apenas o Model Card.
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
    token: str = None,
    readme_only: bool = False
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
    if readme_only:
        print("Modo de Operação:    ⚡ Atualização rápida de Model Card (--readme-only)")
    else:
        print("Modo de Operação:    📦 Upload completo (Pesos + Model Card + GGUF)")
    print("=" * 60)

    # 2. Criar repositório no Hub (se não existir)
    print("1. Verificando/Criando repositório no Hugging Face...")
    create_repo(repo_id=full_repo_id, repo_type="model", exist_ok=True, token=token)
    print("   [✓] Repositório pronto!")

    # 3. Preparar diretório de upload e MODEL_CARD.md como README.md oficial
    os.makedirs(standalone_folder, exist_ok=True)
    dest_readme = os.path.join(standalone_folder, "README.md")
    if os.path.exists(model_card):
        shutil.copy(model_card, dest_readme)
        print(f"   [✓] MODEL_CARD.md copiado como README.md oficial em {dest_readme}")
    else:
        dest_readme = model_card

    # Copiar Modelfile se existir
    for mf_candidate in ["./Modelfile_v1.2", "./Modelfile"]:
        if os.path.exists(mf_candidate):
            shutil.copy(mf_candidate, os.path.join(standalone_folder, "Modelfile"))
            print(f"   [✓] {mf_candidate} copiado para o repositório como Modelfile.")
            break

    # Se for apenas o README, envia em segundos usando api.upload_file
    if readme_only:
        print("\n2. Enviando apenas o README.md atualizado...")
        source_file = dest_readme if os.path.exists(dest_readme) else model_card
        api.upload_file(
            path_or_fileobj=source_file,
            path_in_repo="README.md",
            repo_id=full_repo_id,
            repo_type="model"
        )
        print("   [✓] README.md enviado com sucesso em segundos!")
        print("\n" + "=" * 60)
        print("🎉 Model Card atualizado com sucesso no Hugging Face Hub!")
        print(f"Acesse em: https://huggingface.co/{full_repo_id}")
        print("=" * 60)
        return

    # 4. Incluir GGUF no diretório para quem usa Ollama/llama.cpp
    if gguf_file and os.path.exists(gguf_file):
        dest_gguf = os.path.join(standalone_folder, os.path.basename(gguf_file))
        if not os.path.exists(dest_gguf) and os.path.abspath(gguf_file) != os.path.abspath(dest_gguf):
            print(f"2. Vinculando arquivo GGUF ({gguf_file}) ao repositório...")
            try:
                os.link(gguf_file, dest_gguf) # Hard link economiza espaço em disco
            except Exception:
                shutil.copy(gguf_file, dest_gguf)
        print(f"   [✓] Arquivo GGUF pronto para upload ({os.path.basename(dest_gguf)}).")

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
    parser.add_argument("--repo_name", default="CyberRott-1.2-mini", help="Nome do repositório no Hugging Face")
    parser.add_argument("--folder", default="./CyberRott-Phi4-mini-Standalone", help="Pasta com os pesos standalone")
    parser.add_argument("--gguf", default="./CyberRott-Phi4-mini.gguf", help="Arquivo GGUF para incluir no Hub")
    parser.add_argument("--model_card", default="./MODEL_CARD.md", help="Arquivo Model Card a ser usado como README.md")
    parser.add_argument("--token", default=None, help="Token HF com permissão Write (ou use huggingface-cli login)")
    parser.add_argument("--readme-only", action="store_true", help="Atualiza somente o README.md / Model Card no Hub (rápido, sem subir pesos)")
    args = parser.parse_args()

    upload_model(
        repo_name=args.repo_name,
        standalone_folder=args.folder,
        gguf_file=args.gguf,
        model_card=args.model_card,
        token=args.token,
        readme_only=args.readme_only
    )
