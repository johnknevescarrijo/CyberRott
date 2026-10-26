#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para upload automático do CyberRott (v1.3 GRPO) para o Hugging Face Hub.
Cria o repositório se não existir, copia o MODEL_CARD.md como README.md oficial
e envia os pesos safetensors e/ou arquivo GGUF com suporte a resumo de upload.
Suporta a flag --readme-only para sincronizar rapidamente apenas o Model Card.
Suporta resolução automática de tokens via CLI, HF_TOKEN env, Colab userdata secrets e huggingface_hub cache.
"""

import os
import sys
import shutil
import argparse
from typing import Optional, List
from huggingface_hub import HfApi, create_repo, get_token

DEFAULT_REPO_NAME = "CyberRott-v1.3-mini"

CANDIDATE_FOLDERS: List[str] = [
    "/content/CyberRott-v1.3-BF16-Standalone",
    "./CyberRott-v1.3-Standalone",
    "./CyberRott-v1.3-BF16-Standalone",
    "/content/CyberRott-v1.2-BF16-Standalone",
    "./CyberRott-v1.2-BF16-Standalone",
    "./CyberRott-Phi4-mini-Standalone"
]

CANDIDATE_GGUFS: List[str] = [
    "./CyberRott-v1.3-mini.gguf",
    "/content/CyberRott-v1.3-mini.gguf",
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
    Resolve o token do Hugging Face com precedência hierárquica:
    1. Argumento CLI explícito (--token)
    2. Variável de ambiente (HF_TOKEN ou HUGGING_FACE_HUB_TOKEN)
    3. Google Colab Secrets (userdata.get('HF_TOKEN'))
    4. Cache local do huggingface_hub (login prévio via huggingface-cli)
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
            print("[✓] HF_TOKEN recuperado dos Secrets do Google Colab.")
            return colab_token.strip()
    except Exception:
        pass

    cached_token = get_token()
    if cached_token and cached_token.strip():
        return cached_token.strip()

    return None

def resolve_model_folder(specified_folder: Optional[str] = None) -> str:
    """
    Retorna o primeiro diretório de pesos standalone existente entre os candidatos,
    ou o especificado pelo usuário.
    """
    if specified_folder:
        if os.path.exists(specified_folder):
            return specified_folder
        print(f"[!] Pasta especificada '{specified_folder}' não existe no momento.")
        return specified_folder

    for candidate in CANDIDATE_FOLDERS:
        if os.path.exists(candidate) and os.path.isdir(candidate):
            # Verifica se contém arquivos safetensors ou bin
            files = os.listdir(candidate)
            if any(f.endswith(".safetensors") or f.endswith(".bin") or f == "config.json" for f in files):
                print(f"[✓] Diretório de pesos localizado dinamicamente: {candidate}")
                return candidate

    # Fallback padrão
    fallback = "./CyberRott-v1.3-Standalone"
    print(f"[*] Nenhum diretório pré-existente encontrado. Usando fallback padrão: {fallback}")
    return fallback

def resolve_gguf_file(specified_gguf: Optional[str] = None) -> Optional[str]:
    """Localiza o arquivo GGUF se existente."""
    if specified_gguf and os.path.exists(specified_gguf):
        return specified_gguf

    for candidate in CANDIDATE_GGUFS:
        if os.path.exists(candidate):
            print(f"[✓] Arquivo GGUF localizado dinamicamente: {candidate}")
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
        print("\n❌ ERRO: Token do Hugging Face não encontrado!")
        print("Por favor, forneça o token de uma das seguintes formas:")
        print("  1. No terminal: python scripts/upload_to_hf.py --token hf_xxx")
        print("  2. Variável de ambiente: export HF_TOKEN=hf_xxx")
        print("  3. No Google Colab: Adicione 'HF_TOKEN' nos Secrets (ícone de chave no menu lateral esquerdo)")
        print("  4. Login local via CLI: huggingface-cli login")
        sys.exit(1)

    api = HfApi(token=resolved_token)
    
    # 1. Identificar usuário autenticado
    try:
        user_info = api.whoami()
        username = user_info["name"]
    except Exception as e:
        print(f"❌ Falha ao validar token do Hugging Face: {e}")
        sys.exit(1)

    full_repo_id = f"{username}/{repo_name}"
    
    print("=" * 65)
    print("🛡️  CyberRott v1.3 GRPO — Publicação no Hugging Face Hub")
    print(f"Usuário autenticado: {username}")
    print(f"Repositório Alvo:    https://huggingface.co/{full_repo_id}")
    if readme_only:
        print("Modo de Operação:    ⚡ Atualização rápida de Model Card (--readme-only)")
    else:
        print("Modo de Operação:    📦 Upload completo (Pesos + Model Card + Artefatos)")
    print("=" * 65)

    # 2. Criar repositório no Hub (se não existir)
    print("1. Verificando/Criando repositório no Hugging Face...")
    create_repo(repo_id=full_repo_id, repo_type="model", exist_ok=True, token=resolved_token)
    print("   [✓] Repositório confirmado e acessível!")

    # 3. Preparar diretório de upload e MODEL_CARD.md como README.md oficial
    target_folder = resolve_model_folder(standalone_folder)
    os.makedirs(target_folder, exist_ok=True)
    dest_readme = os.path.join(target_folder, "README.md")
    
    if os.path.exists(model_card):
        shutil.copy(model_card, dest_readme)
        print(f"   [✓] MODEL_CARD.md copiado como README.md oficial em: {dest_readme}")
    else:
        print(f"   [!] Aviso: {model_card} não encontrado. Buscando README.md local...")
        dest_readme = os.path.join(target_folder, "README.md")

    # Copiar Modelfile se existir
    for mf_candidate in CANDIDATE_MODELFILES:
        if os.path.exists(mf_candidate):
            shutil.copy(mf_candidate, os.path.join(target_folder, "Modelfile"))
            print(f"   [✓] {mf_candidate} copiado para o repositório como Modelfile.")
            break

    # Se for apenas o README, envia em segundos usando api.upload_file
    if readme_only:
        print("\n2. Enviando apenas o README.md / Model Card atualizado...")
        source_file = dest_readme if os.path.exists(dest_readme) else model_card
        api.upload_file(
            path_or_fileobj=source_file,
            path_in_repo="README.md",
            repo_id=full_repo_id,
            repo_type="model"
        )
        print("   [✓] README.md atualizado com sucesso no Hugging Face!")
        print("\n" + "=" * 65)
        print("🎉 Model Card atualizado com sucesso no Hugging Face Hub!")
        print(f"Acesse em: https://huggingface.co/{full_repo_id}")
        print("=" * 65)
        return

    # 4. Incluir GGUF no diretório para quem usa Ollama/llama.cpp (se disponível)
    resolved_gguf = resolve_gguf_file(gguf_file)
    if resolved_gguf and os.path.exists(resolved_gguf):
        dest_gguf = os.path.join(target_folder, os.path.basename(resolved_gguf))
        if not os.path.exists(dest_gguf) and os.path.abspath(resolved_gguf) != os.path.abspath(dest_gguf):
            print(f"2. Vinculando arquivo GGUF ({resolved_gguf}) ao repositório...")
            try:
                os.link(resolved_gguf, dest_gguf)  # Hard link economiza espaço em disco
            except Exception:
                shutil.copy(resolved_gguf, dest_gguf)
        print(f"   [✓] Arquivo GGUF pronto para upload ({os.path.basename(dest_gguf)}).")
    else:
        print("2. [INFO] Nenhum binário GGUF detectado localmente. Prosseguindo com pesos BF16 / Safetensors.")

    # 5. Fazer upload da pasta inteira com progresso e retry automático
    print(f"\n3. Iniciando upload da pasta de pesos ({target_folder})...")
    print("   Isso pode levar alguns minutos dependendo do tamanho e da largura de banda.")
    
    api.upload_folder(
        folder_path=target_folder,
        repo_id=full_repo_id,
        repo_type="model"
    )

    print("\n" + "=" * 65)
    print("🎉 SUCESSO! O CyberRott v1.3 GRPO está publicado no Hugging Face Hub!")
    print(f"Acesse o repositório em: https://huggingface.co/{full_repo_id}")
    print("Comandos de execução recomendados:")
    print(f"   • Transformers / vLLM: vllm serve {full_repo_id}")
    print(f"   • Ollama Hub:         ollama run hf.co/{full_repo_id}")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload do CyberRott v1.3 para o Hugging Face Hub")
    parser.add_argument("--repo_name", default=DEFAULT_REPO_NAME, help="Nome do repositório no Hugging Face (default: CyberRott-v1.3-mini)")
    parser.add_argument("--folder", default=None, help="Pasta com os pesos standalone (auto-resolvida se omitida)")
    parser.add_argument("--gguf", default=None, help="Arquivo GGUF para incluir no Hub (opcional)")
    parser.add_argument("--model_card", default="./MODEL_CARD.md", help="Arquivo Model Card a ser usado como README.md")
    parser.add_argument("--token", default=None, help="Token HF com permissão Write (ou use Colab Secret / HF_TOKEN env)")
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
