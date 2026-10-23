#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline de Exportação e Quantização GGUF para CyberRott 1.2
============================================================
Automatiza o fluxo completo pós-treinamento para deploy local no Ollama / llama.cpp:
1. Fusão Sequencial de Adaptadores (Base Model + SFT LoRA + GRPO RL -> Standalone).
2. Download e Configuração do utilitário llama.cpp.
3. Conversão de pesos HuggingFace para GGUF base (F16).
4. Quantização de alta performance (padrão: Q4_K_M para ~2.3 GB ou Q8_0).
5. Geração automática do arquivo Modelfile pronto para execução imediata no Ollama.

Exemplo de uso:
    python scripts/export_gguf.py \
        --sft_adapter /content/drive/MyDrive/CyberRott_Checkpoints/cyberrott_v1_1_sft_lora \
        --grpo_adapter /content/drive/MyDrive/CyberRott_Checkpoints/CyberRott_v1_2_GRPO \
        --quant_type Q4_K_M \
        --output_dir ./exports
"""

import os
import sys

# Desativa TensorFlow quebrado no Python 3.13 do Colab
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
sys.modules["tensorflow"] = None

import shutil
import argparse
import subprocess
from pathlib import Path

# Desativa verificação do torchao para compatibilidade no Google Colab
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass


def run_cmd(command: str, description: str = ""):
    """Executa um comando no shell com tratamento de erro e logs visuais."""
    if description:
        print(f"\n[*] {description}...")
    print(f"    $ {command}")
    res = subprocess.run(command, shell=True, text=True)
    if res.returncode != 0:
        print(f"[!] Erro ao executar: {command}")
        sys.exit(res.returncode)


def export_pipeline(
    base_model: str,
    sft_adapter: str,
    grpo_adapter: str,
    quant_type: str,
    output_dir: str,
    torch_dtype: str = "bfloat16",
    ollama_model_name: str = "cyberrott-v1.2",
    skip_merge: bool = False,
    standalone_path: str = None,
):
    print("=" * 70)
    print("🛡️  CyberRott v1.2 — Pipeline Automatizado de Exportação & Quantização GGUF")
    print("=" * 70)
    print(f"Modelo Base:      {base_model}")
    print(f"Adaptador SFT:    {sft_adapter or 'Nenhum'}")
    print(f"Adaptador GRPO:   {grpo_adapter}")
    print(f"Tipo Quantização: {quant_type}")
    print(f"Diretório Destino: {output_dir}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    standalone_dir = standalone_path or os.path.join(output_dir, "CyberRott-v1.2-Standalone")
    f16_gguf_path = os.path.join(output_dir, "cyberrott-v1.2-f16.gguf")
    quant_gguf_path = os.path.join(output_dir, f"cyberrott-v1.2-{quant_type.lower()}.gguf")
    modelfile_path = os.path.join(output_dir, "Modelfile")
    llama_cpp_dir = os.path.join(output_dir, "llama.cpp")

    # -------------------------------------------------------------------------
    # Passo 1: Fusão dos Pesos (Merge & Unload)
    # -------------------------------------------------------------------------
    if not skip_merge:
        print("\n" + "=" * 70)
        print("ETAPA 1: FUSÃO SEQUENCIAL DE ADAPTADORES (MERGE & UNLOAD)")
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
        print(f"\n[*] Pulando etapa de fusão. Utilizando modelo standalone existente em: {standalone_dir}")

    # -------------------------------------------------------------------------
    # Passo 2: Clonar e Configurar llama.cpp
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ETAPA 2: CONFIGURAÇÃO DO ENGINE LLAMA.CPP")
    print("=" * 70)

    if not os.path.exists(llama_cpp_dir):
        run_cmd(
            f"git clone --depth 1 https://github.com/ggerganov/llama.cpp.git {llama_cpp_dir}",
            "Clonando repositório llama.cpp"
        )
    else:
        print(f"[✓] Repositório llama.cpp já presente em: {llama_cpp_dir}")

    # Instala requisitos de conversão do llama.cpp
    run_cmd(
        f"{sys.executable} -m pip install -q -r {llama_cpp_dir}/requirements.txt",
        "Instalando dependências de conversão do llama.cpp"
    )

    # -------------------------------------------------------------------------
    # Passo 3: Conversão HuggingFace -> GGUF F16
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ETAPA 3: CONVERSÃO DE PESOS PARA FORMATO GGUF (F16)")
    print("=" * 70)

    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        # Fallback para repositório recente do llama.cpp
        convert_script = os.path.join(llama_cpp_dir, "convert-hf-to-gguf.py")

    run_cmd(
        f"{sys.executable} {convert_script} {standalone_dir} --outtype f16 --outfile {f16_gguf_path}",
        "Convertendo tensores HuggingFace para arquivo GGUF F16"
    )

    # -------------------------------------------------------------------------
    # Passo 4: Quantização de Alta Eficiência (ex: Q4_K_M)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"ETAPA 4: COMPILAÇÃO E QUANTIZAÇÃO PARA {quant_type}")
    print("=" * 70)

    quantize_bin = os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize")
    if not os.path.exists(quantize_bin):
        quantize_bin_alt = os.path.join(llama_cpp_dir, "llama-quantize")
        if os.path.exists(quantize_bin_alt):
            quantize_bin = quantize_bin_alt
        else:
            build_dir = os.path.join(llama_cpp_dir, "build")
            run_cmd(f"cmake -B {build_dir} {llama_cpp_dir}", "Configurando CMake para compilação")
            run_cmd(f"cmake --build {build_dir} --config Release -j --target llama-quantize", "Compilando binário llama-quantize")

    run_cmd(
        f"{quantize_bin} {f16_gguf_path} {quant_gguf_path} {quant_type}",
        f"Executando quantização {quant_type} no modelo"
    )

    # -------------------------------------------------------------------------
    # Passo 5: Geração Automática do Modelfile para Ollama
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ETAPA 5: GERAÇÃO DO MODELFILE PARA OLLAMA")
    print("=" * 70)

    modelfile_content = f"""FROM {os.path.abspath(quant_gguf_path)}

TEMPLATE \"\"\"<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"

SYSTEM \"\"\"Você é o CyberRott, uma IA especialista em segurança ofensiva, defensiva e análise de vulnerabilidades com raciocínio analítico profundo de 7 estágios.\"\"\"

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|end|>"
PARAMETER stop "<|endoftext|>"
"""

    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    final_size_mb = os.path.getsize(quant_gguf_path) / (1024 * 1024)
    print(f"[✓] Modelfile gerado com sucesso em: {modelfile_path}")
    print(f"[✓] Binário quantizado final:        {quant_gguf_path} ({final_size_mb:.1f} MB / {final_size_mb/1024:.2f} GB)")

    print("\n" + "=" * 70)
    print("🎉 EXPORTAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 70)
    print("Como rodar localmente no seu Ollama agora:")
    print(f"  1. Entre na pasta:   cd {os.path.abspath(output_dir)}")
    print(f"  2. Registre no Ollama: ollama create {ollama_model_name} -f Modelfile")
    print(f"  3. Execute o modelo:   ollama run {ollama_model_name}")
    print("=" * 70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline automatizado de Fusão LoRA + Conversão GGUF + Quantização para CyberRott 1.2"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/Phi-4-mini-instruct",
        help="Caminho do modelo base Hugging Face (padrão: microsoft/Phi-4-mini-instruct)."
    )
    parser.add_argument(
        "--sft_adapter",
        type=str,
        default="./cyberrott_v1_1_sft_lora",
        help="Pasta do adaptador supervisionado da Fase 2 (opcional, pode ser omitido se None)."
    )
    parser.add_argument(
        "--grpo_adapter", "--adapter",
        dest="grpo_adapter",
        type=str,
        default="./CyberRott-v1.2-GRPO",
        help="Pasta do adaptador LoRA treinado com GRPO na Fase 3."
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="Q4_K_M",
        choices=["Q4_K_M", "Q4_0", "Q5_K_M", "Q8_0", "f16"],
        help="Método de quantização para deploy no Ollama/llama.cpp (padrão: Q4_K_M)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./CyberRott-v1.2-Export",
        help="Diretório onde serão salvos o modelo standalone, os GGUFs e o Modelfile."
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Precisão numérica para o merge dos pesos (padrão: bfloat16)."
    )
    parser.add_argument(
        "--ollama_model_name",
        type=str,
        default="cyberrott-v1.2",
        help="Nome que será usado no comando 'ollama run' (padrão: cyberrott-v1.2)."
    )
    parser.add_argument(
        "--skip_merge",
        action="store_true",
        help="Pula o merge se você já possuir os pesos standalone salvos em disco."
    )
    parser.add_argument(
        "--standalone_path",
        type=str,
        default=None,
        help="Caminho alternativo para o modelo standalone já fundido (usado se --skip_merge)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Normalizar sft_adapter se não existir
    sft_path = args.sft_adapter
    if sft_path and not os.path.exists(sft_path):
        # Tenta checar variações comuns
        drive_candidate = "/content/drive/MyDrive/CyberRott_Checkpoints/cyberrott_v1_1_sft_lora"
        if os.path.exists(drive_candidate):
            sft_path = drive_candidate
        else:
            sft_path = None

    grpo_path = args.grpo_adapter
    if not os.path.exists(grpo_path):
        drive_grpo = "/content/drive/MyDrive/CyberRott_Checkpoints/CyberRott_v1_2_GRPO"
        if os.path.exists(drive_grpo):
            grpo_path = drive_grpo

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
