#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Fusão e Preparação de Dados para o CyberRott 1.0.
Executa Anchor Mixing (Amostras Complexas de Ouro com Upsampling) +
Streaming e Filtragem de Cibersegurança do GooseReason (NVIDIA) + Shuffling.
"""

import os
import re
import json
import random
import argparse
from typing import List, Dict, Any

# Lista aprofundada de termos de cibersegurança, rede, criptografia e sistemas operacionais
CYBER_KEYWORDS = [
    # Vulnerabilidades e Exploração
    "vulnerability", "exploit", "cve", "cve-", "buffer overflow", "stack overflow",
    "heap overflow", "use-after-free", "out-of-bounds", "format string",
    "race condition", "toctou", "pwn", "shellcode", "rop chain", "ret2libc",
    "privilege escalation", "bypass", "taint analysis", "injection",
    "sql injection", "sqli", "cross-site scripting", "xss", "csrf", "ssrf",
    "command injection", "path traversal", "directory traversal",
    # Engenharia Reversa e Binários
    "reverse engineering", "disassembly", "decompiler", "decompilation",
    "gdb", "radare2", "ghidra", "ida pro", "binwalk", "objdump",
    "elf", "pe binary", "assembly", "x86", "x86_64", "arm64",
    # Malware e Forense
    "malware", "ransomware", "backdoor", "trojan", "rootkit", "keylogger",
    "botnet", "forensics", "memory dump", "volatility", "sysmon",
    "wireshark", "pcap", "packet analysis", "network traffic", "yara", "sigma",
    # Redes e Protocolos
    "tcp/ip", "packet", "dns", "dns tunneling", "socket", "http request",
    "firewall", "ids", "ips", "edr", "mitre", "port scan", "nmap",
    # Criptografia e Autenticação
    "cryptography", "rsa", "aes", "elliptic curve", "diffie-hellman",
    "sha256", "jwt", "hash collision", "nonce reuse", "kerberos",
    "active directory", "golden ticket", "silver ticket", "pass-the-hash",
    "authentication", "authorization", "sandbox", "kernel exploit", "container escape"
]

def compile_cyber_regex(keywords: List[str]) -> re.Pattern:
    """Compila regex com word-boundary para termos curtos (evitando falsos positivos)."""
    patterns = []
    for kw in keywords:
        if len(kw) <= 4:
            patterns.append(rf"\b{re.escape(kw)}\b")
        else:
            patterns.append(re.escape(kw))
    return re.compile("|".join(patterns), re.IGNORECASE)

def load_anchor_samples(anchor_file: str) -> List[Dict[str, Any]]:
    """Carrega as amostras âncora no formato padronizado ChatML."""
    if not os.path.exists(anchor_file):
        print(f"[Aviso] Arquivo âncora '{anchor_file}' não encontrado.")
        return []

    samples = []
    with open(anchor_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "messages" in data and isinstance(data["messages"], list):
                    samples.append(data)
                elif "instruction" in data and ("final_answer" in data or "reasoning_trace" in data):
                    # Conversão de formato de raciocínio bruto
                    instruction = data.get("instruction", "").strip()
                    user_input = data.get("input", "").strip()
                    reasoning = data.get("reasoning_trace", "").strip()
                    final_answer = data.get("final_answer", "").strip()

                    prompt = f"{instruction}\n\n```\n{user_input}\n```" if user_input else instruction
                    if reasoning and not reasoning.startswith("<think>"):
                        reasoning = f"<think>\n{reasoning}"
                    if reasoning and not reasoning.endswith("</think>"):
                        reasoning = f"{reasoning}\n</think>"
                    assistant_content = f"{reasoning}\n\n{final_answer}" if reasoning else final_answer

                    samples.append({
                        "messages": [
                            {"role": "system", "content": "Você é o CyberRott, uma IA especialista em segurança ofensiva, defensiva e análise forense com raciocínio analítico profundo."},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": assistant_content}
                        ]
                    })
            except Exception as e:
                print(f"[Aviso] Falha ao processar linha {idx} do arquivo âncora: {e}")
                continue

    return samples

def extract_qa_from_hf_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Inspeciona dinamicamente as colunas de um registro do Hugging Face e formata em ChatML."""
    # Caso 1: Item já estruturado em messages
    if "messages" in item and isinstance(item["messages"], list):
        return {"messages": item["messages"]}

    # Caso 2: Item em formato de conversação (ex: ShareGPT / Vicuna)
    if "conversations" in item and isinstance(item["conversations"], list):
        role_map = {"human": "user", "gpt": "assistant", "chatgpt": "assistant", "system": "system"}
        msgs = []
        for c in item["conversations"]:
            role = role_map.get(c.get("from", "").lower(), "user")
            msgs.append({"role": role, "content": str(c.get("value", ""))})
        if msgs:
            return {"messages": msgs}

    # Caso 3: Campos clássicos de Pergunta / Problema
    question = (
        item.get("problem")
        or item.get("question")
        or item.get("prompt")
        or item.get("instruction")
        or item.get("input")
        or item.get("text")
    )

    # Identificar resposta (suporte a opções do GooseReason-0.7M)
    options = item.get("options")
    answer_raw = item.get("answer")
    solution_text = None

    if options and isinstance(options, list) and answer_raw:
        ans_str = str(answer_raw).strip().upper()
        if len(ans_str) == 1 and "A" <= ans_str <= "Z":
            idx = ord(ans_str) - ord("A")
            if 0 <= idx < len(options):
                solution_text = f"Opção correta: **{ans_str}**\n\n```\n{options[idx]}\n```"
            else:
                solution_text = f"Opção correta: \\boxed{{{ans_str}}}"

    if not solution_text:
        solution_text = (
            item.get("solution")
            or item.get("answer")
            or item.get("response")
            or item.get("output")
        )

    if question and solution_text:
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "Você é o CyberRott, uma IA especialista em segurança ofensiva, defensiva e análise forense com raciocínio analítico profundo."
                },
                {"role": "user", "content": str(question).strip()},
                {"role": "assistant", "content": str(solution_text).strip()}
            ]
        }

    return None

def fetch_hf_cyber_samples(
    dataset_name: str,
    max_samples: int,
    cyber_regex: re.Pattern
) -> List[Dict[str, Any]]:
    """Carrega via streaming o dataset do Hugging Face e filtra amostras de cibersegurança."""
    hf_samples = []
    print(f"[*] Conectando ao Hugging Face para streaming do dataset '{dataset_name}'...")

    try:
        from datasets import load_dataset
    except ImportError:
        print("[Aviso] Biblioteca 'datasets' não encontrada. Instalando ou ignorando streaming do HF.")
        return []

    # Tenta carregar os splits disponíveis (train, code, stem, math)
    ds = None
    candidate_splits = ["train", "code", "stem", "math"]
    for split_choice in candidate_splits:
        try:
            ds = load_dataset(dataset_name, split=split_choice, streaming=True)
            print(f"[✓] Streaming iniciado com sucesso via split='{split_choice}'.")
            break
        except Exception:
            continue

    if ds is None:
        try:
            ds_dict = load_dataset(dataset_name, streaming=True)
            active_split = "code" if "code" in ds_dict else list(ds_dict.keys())[0]
            ds = ds_dict[active_split]
            print(f"[✓] Streaming iniciado via split dinâmico='{active_split}'.")
        except Exception as e:
            print(f"[Aviso] Não foi possível inicializar streaming do dataset '{dataset_name}': {e}")
            return []

    count = 0
    checked = 0
    try:
        for item in ds:
            checked += 1
            # Montar texto unificado para verificação de keywords
            text_parts = []
            for k, v in item.items():
                if isinstance(v, (str, int, float)):
                    text_parts.append(str(v))
                elif isinstance(v, list):
                    for sub in v:
                        if isinstance(sub, str):
                            text_parts.append(sub)
                        elif isinstance(sub, dict):
                            text_parts.extend(str(val) for val in sub.values() if isinstance(val, (str, int, float)))
            full_text = " ".join(text_parts)

            if cyber_regex.search(full_text):
                formatted = extract_qa_from_hf_item(item)
                if formatted:
                    hf_samples.append(formatted)
                    count += 1
                    if count % 100 == 0:
                        print(f"[*] GooseReason-Cyber filtrado: {count}/{max_samples} amostras (analisadas {checked})...")
                    if count >= max_samples:
                        break
    except Exception as e:
        print(f"[Aviso] Interrupção durante o streaming de '{dataset_name}': {e}")

    print(f"[✓] GooseReason-Cyber concluído: {len(hf_samples)} amostras coletadas.")
    return hf_samples

def run_fusion_pipeline(
    anchor_file: str = "dataset_cyber_ready.jsonl",
    anchor_repeat: int = 20,
    hf_dataset: str = "nvidia/Nemotron-Research-GooseReason-0.7M",
    max_hf_samples: int = 2500,
    output_file: str = "cyberrott_final_train.jsonl",
    seed: int = 42
):
    """Executa a fusão completa: Âncoras ponderadas + GooseReason filtrado + Shuffling."""
    print("=" * 70)
    print("🛡️  CyberRott 1.0 - Pipeline de Fusão de Dados de Treinamento")
    print("=" * 70)
    print(f"• Arquivo Âncora: {anchor_file} (Upsampling: {anchor_repeat}x)")
    print(f"• Dataset Hugging Face: {hf_dataset} (Máximo: {max_hf_samples})")
    print(f"• Arquivo de Saída: {output_file}")
    print(f"• Random Seed: {seed}")
    print("-" * 70)

    # 1. Carregar Amostras Âncora e Aplicar Upsampling
    raw_anchors = load_anchor_samples(anchor_file)
    unique_anchor_count = len(raw_anchors)
    anchor_samples = raw_anchors * anchor_repeat
    total_anchor_count = len(anchor_samples)
    print(f"[✓] Âncoras carregadas: {unique_anchor_count} únicas -> {total_anchor_count} após upsampling ({anchor_repeat}x).")

    # 2. Carregar e Filtrar Dataset do Hugging Face (com try/except elegante)
    cyber_regex = compile_cyber_regex(CYBER_KEYWORDS)
    hf_cyber_samples = []
    if max_hf_samples > 0 and hf_dataset:
        try:
            hf_cyber_samples = fetch_hf_cyber_samples(
                dataset_name=hf_dataset,
                max_samples=max_hf_samples,
                cyber_regex=cyber_regex
            )
        except Exception as e:
            print(f"[Aviso] Falha de conexão ou erro no Hugging Face: {e}")
            print("[Info] Continuando o pipeline preservando os exemplos âncora.")

    total_hf_count = len(hf_cyber_samples)

    # 3. Combinar e Embaralhar (Anchor Mixing + Shuffling)
    combined_samples = anchor_samples + hf_cyber_samples
    if not combined_samples:
        print("[Erro Crítico] Nenhuma amostra disponível para compor o dataset de treinamento!")
        return

    random.seed(seed)
    random.shuffle(combined_samples)

    # 4. Salvar Dataset Final
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f_out:
        for entry in combined_samples:
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 5. Relatório Consolidado
    print("=" * 70)
    print("📊 RELATÓRIO CONSOLIDADO DO DATASET DE TREINAMENTO")
    print("=" * 70)
    print(f"• Total de Âncoras de Ouro (únicas):        {unique_anchor_count}")
    print(f"• Fator de Upsampling das Âncoras:          {anchor_repeat}x")
    print(f"• Total de Amostras Âncora no Treino:       {total_anchor_count}")
    print(f"• Total de Amostras GooseReason-Cyber:      {total_hf_count}")
    print(f"• TOTAL FINAL COMBINADO (com Shuffling):    {len(combined_samples)}")
    print(f"• Arquivo Gerado:                           {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Fusão de Dados para Fine-Tuning do CyberRott 1.0")
    parser.add_argument("--anchor_file", default="dataset_cyber_ready.jsonl", help="Caminho do arquivo com os 16 exemplos complexos de ouro")
    parser.add_argument("--anchor_repeat", type=int, default=20, help="Fator de repetição (upsampling) para as amostras âncora (default: 20 -> 320 amostras)")
    parser.add_argument("--hf_dataset", default="nvidia/Nemotron-Research-GooseReason-0.7M", help="Dataset no Hugging Face para streaming e filtragem")
    parser.add_argument("--max_hf_samples", type=int, default=2500, help="Quantidade máxima de amostras de segurança a extrair do Hugging Face")
    parser.add_argument("--output", default="cyberrott_final_train.jsonl", help="Caminho do arquivo JSONL de saída final")
    parser.add_argument("--seed", type=int, default=42, help="Seed para reproduzibilidade do embaralhamento (shuffling)")
    args = parser.parse_args()

    run_fusion_pipeline(
        anchor_file=args.anchor_file,
        anchor_repeat=args.anchor_repeat,
        hf_dataset=args.hf_dataset,
        max_hf_samples=args.max_hf_samples,
        output_file=args.output,
        seed=args.seed
    )
