#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CyberRott 1.1 - Pipeline Unificado de Fusão e Preparação de Dados
(Pilar 1: Zero-API Cost Data Expansion - 100% Open-Source & Community Tiers).

Funcionalidades:
1. Anchor Mixing: Amostras Complexas de Ouro com Upsampling ponderado.
2. Catálogo Oficial v1.1: Streaming multi-dataset de corpora abertos do Hugging Face:
   - nvidia/Nemotron-Research-GooseReason-0.7M (Raciocínio lógico & código)
   - m-a-p/CodeFeedback-Filtered-Instruction (Auditoria de código, memória & refatoração)
   - open-r1/codeforces-cots / bespokelabs/Bespoke-Stratos-17k (Trajetórias long-CoT)
   - CyberNative/Code_Vulnerability_Security_DPO (Detecção de vulnerabilidades & patches DPO)
   - AlicanKiraz0/Cybersecurity-Dataset-v1 (Red/Blue Team, MITRE ATT&CK & Forense)
3. Scrubber Programático de Recusas de Alinhamento e Remoção de Disclaimers Moralistas.
4. Formatador Automático para CoT Nativo de 7 Estágios (<think>...</think>).
5. Desduplicação Determinística por Hash SHA-256 do Prompt de Usuário.
"""

import os
import re
import sys
import json
import random
import hashlib
import argparse
from typing import List, Dict, Any, Optional, Set, Tuple

# =============================================================================
# 1. CATÁLOGO OFICIAL DE DATASETS OPEN-SOURCE CYBERROTT v1.1 (ZERO-API COST)
# =============================================================================

CYBERROTT_V1_1_DATASETS: Dict[str, Dict[str, Any]] = {
    "goose_reason": {
        "repo_id": "nvidia/Nemotron-Research-GooseReason-0.7M",
        "name": "NVIDIA GooseReason-0.7M",
        "target_samples": 7500,
        "requires_cyber_filter": True,
        "domain": "Deep Reasoning & Code Logic",
        "description": "Corpus principal de raciocínio analítico da NVIDIA com cadeias de dedução formal.",
        "splits": ["train", "code", "stem", "math"],
    },
    "code_feedback": {
        "repo_id": "m-a-p/CodeFeedback-Filtered-Instruction",
        "name": "CodeFeedback Filtered Instruction",
        "target_samples": 1000,
        "requires_cyber_filter": True,
        "domain": "Code Audit, Memory & Refactoring",
        "description": "Trajetórias de revisão de código, correção de bugs de memória e refatoração segura.",
        "splits": ["train"],
    },
    "open_r1": {
        "repo_id": "open-r1/codeforces-cots",
        "fallback_repo_id": "bespokelabs/Bespoke-Stratos-17k",
        "name": "Open-R1 CoTs (Fallback: Bespoke-Stratos)",
        "target_samples": 800,
        "requires_cyber_filter": True,
        "domain": "Long-CoT Reasoning & Trajectories",
        "description": "Trajetórias abertas de raciocínio comunitário com reflexão longa estruturada.",
        "splits": ["train"],
    },
    "cyber_native": {
        "repo_id": "CyberNative/Code_Vulnerability_Security_DPO",
        "name": "CyberNative Vulnerability Security DPO",
        "target_samples": 500,
        "requires_cyber_filter": False,
        "domain": "Code Vulnerabilities & CWE DPO Patches",
        "description": "Pares de código vulnerável (rejected) vs seguro (chosen) com classificação CWE.",
        "splits": ["train"],
    },
    "cybersecurity_v1": {
        "repo_id": "AlicanKiraz0/Cybersecurity-Dataset-v1",
        "name": "AlicanKiraz0 Cybersecurity Dataset v1",
        "target_samples": 400,
        "requires_cyber_filter": False,
        "domain": "Red/Blue Team, MITRE ATT&CK & Forensics",
        "description": "Dataset especializado em táticas ofensivas, defensivas, CVEs e forense digital.",
        "splits": ["train"],
    },
}

# =============================================================================
# 2. LISTA PROFUNDA DE TERMOS DE CIBERSEGURANÇA E SISTEMAS OPERACIONAIS
# =============================================================================

CYBER_KEYWORDS: List[str] = [
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

# =============================================================================
# 3. SCRUBBER DE RECUSAS E DISCLAIMERS MORALISTAS
# =============================================================================

# Padrões explícitos de recusa de alinhamento conversacional
REFUSAL_PATTERNS: List[str] = [
    r"\bas an ai\b",
    r"\bas a language model\b",
    r"\bi cannot (?:assist|fulfill|comply|help|provide|generate|participate|create|condone)\b",
    r"\bi can't (?:assist|fulfill|comply|help|provide|generate|participate|create|condone)\b",
    r"\bi am unable to (?:assist|fulfill|comply|help|provide|generate|participate|create|condone)\b",
    r"\bi'm unable to (?:assist|fulfill|comply|help|provide|generate|participate|create|condone)\b",
    r"\bi apologize,?\s+but\s+i cannot\b",
    r"\bi'm sorry,?\s+but\s+i cannot\b",
    r"\bi am not able to fulfill this request\b",
    r"\bmy safety guidelines\b",
    r"\bviolates? (?:our|my|safety) policies\b",
    r"\bcannot provide (?:instructions|code|exploits?|assistance) for (?:malicious|illegal|harmful|unauthorized)\b",
    r"\bi cannot create (?:malware|exploits?|payloads?|viruses?)\b",
    r"\bi must decline this request\b",
    r"\bresponsible disclosure prohibits me from\b",
]

# Disclaimers moralizantes ou sermões desnecessários
DISCLAIMER_PATTERNS: List[str] = [
    r"(?im)^(?:disclaimer|note|warning|aviso|atenção|important notice)[\s\:\-]+.*?(?:educational purposes?|authorized (?:testing|systems?|environments?)|without (?:prior )?(?:written )?permission|illegal|fins educacionais).*?\n+",
    r"(?im)\n+(?:disclaimer|note|warning|aviso)[\s\:\-]+.*?(?:educational purposes?|authorized (?:testing|systems?|environments?)|without (?:prior )?(?:written )?permission|illegal|fins educacionais).*?$",
    r"(?im)\b(?:please note that|remember that|keep in mind that)\s+(?:hacking without permission|unauthorized access to computer systems|attacking systems)\s+is illegal.*?\n*",
    r"(?im)\bthis information is strictly for educational and defensive purposes only\.?\b",
    r"(?im)\balways ensure you have explicit (?:written )?permission before (?:testing|accessing).*?\b",
    r"(?im)\bdisclaimer:\s*i cannot condone malicious activities\.?\b",
]

def is_conversational_refusal(text: str) -> bool:
    """Verifica se a resposta do modelo consiste substancialmente em recusa de alinhamento."""
    if not text:
        return True

    clean_text = text.strip()

    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, clean_text, re.IGNORECASE):
            # Se for resposta curta (< 600 caracteres) com marcador de recusa: descarta
            if len(clean_text) < 600:
                return True
            # Se começa com recusa declarada e não possui código técnico formatado: descarta
            if re.search(r"^(?:i cannot|as an ai|i'm sorry|i apologize)", clean_text, re.IGNORECASE) and "```" not in clean_text:
                return True
            # Se possui apenas poucas linhas declaratórias de recusa
            lines = [l.strip() for l in clean_text.split("\n") if l.strip()]
            if len(lines) <= 4:
                return True

    return False

def clean_moralizing_disclaimers(text: str) -> str:
    """Remove sermões moralistas e disclaimers vazios no início ou fim de respostas técnicas."""
    if not text:
        return text

    cleaned = text
    for pattern in DISCLAIMER_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned).strip()

    # Normalizar quebras de linha excessivas
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned

def scrub_refusal_and_boilerplate(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Aplica o scrubber completo em uma amostra no formato ChatML.
    Retorna a amostra limpa ou None se a resposta for uma recusa irrecuperável.
    """
    messages = sample.get("messages", [])
    if not messages:
        return None

    cleaned_messages = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "assistant":
            if is_conversational_refusal(content):
                return None
            content = clean_moralizing_disclaimers(content)
            if not content.strip():
                return None

        cleaned_messages.append({"role": role, "content": content})

    result = {k: v for k, v in sample.items() if k != "messages"}
    result["messages"] = cleaned_messages
    return result

# =============================================================================
# 4. FORMATADOR DE RACIOCÍNIO CoT 7-ESTÁGIOS (<think>...</think>)
# =============================================================================

def format_to_cyberrott_cot(text: str, enforce_think: bool = True) -> str:
    """
    Normaliza tags alternativas (<|begin_of_thought|>, <thought>, <reasoning>)
    para o formato nativo <think>...</think> do CyberRott e estrutura respostas sem CoT.
    """
    if not text:
        return text

    # 1. Normalizar tags alternativas
    normalized = text
    normalized = re.sub(r"<\s*\|begin_of_thought\|\s*>", "<think>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"<\s*\|end_of_thought\|\s*>", "</think>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"<\s*thought\s*>", "<think>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"<\s*/\s*thought\s*>", "</think>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"<\s*reasoning\s*>", "<think>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"<\s*/\s*reasoning\s*>", "</think>", normalized, flags=re.IGNORECASE)

    # 2. Se já possui tag <think> e </think> fechada, manter
    if "<think>" in normalized and "</think>" in normalized:
        return normalized

    # 3. Se possui abertura sem fechamento, fechar adequadamente
    if "<think>" in normalized and "</think>" not in normalized:
        return normalized + "\n</think>"

    # 4. Se não possui tag <think> e enforce_think estiver ativo
    if enforce_think:
        # Tenta separar por cabeçalhos típicos de resposta final
        split_patterns = [
            r"\n(?=###\s+(?:Solution|Final Answer|Code|Correction|Remediation|Answer))",
            r"\n(?=\*\*Final Answer\*\*:?)",
            r"\n(?=Final Answer:)",
            r"\n(?=Resposta Final:)",
            r"\n(?=FATO:)",
            r"\n(?=```(?:python|c|cpp|bash|json|sql|go|java|php))",
        ]

        for pattern in split_patterns:
            parts = re.split(pattern, normalized, maxsplit=1)
            if len(parts) == 2 and len(parts[0].strip()) > 40 and len(parts[1].strip()) > 20:
                reasoning = parts[0].strip()
                final_answer = parts[1].strip()
                return f"<think>\n{reasoning}\n</think>\n\n{final_answer}"

        # Se não há cabeçalho explícito, divide entre premissa técnica e solução
        paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
        if len(paragraphs) >= 2:
            reasoning = paragraphs[0]
            conclusion = "\n\n".join(paragraphs[1:])
            return f"<think>\n**Análise Técnica**: {reasoning}\n</think>\n\n{conclusion}"
        else:
            return f"<think>\n**Análise Preliminar**: Avaliação de parâmetros e mecânica do problema.\n</think>\n\n{normalized}"

    return normalized

def format_cybernative_dpo_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Sintetiza deterministicamente uma amostra de auditoria no formato 7-estágios
    do CyberRott a partir dos metadados de vulnerabilidade (DPO: chosen vs rejected).
    """
    vulnerability = (
        item.get("vulnerability")
        or item.get("vuln_type")
        or item.get("cwe")
        or "Vulnerabilidade de Segurança de Código"
    )
    lang = item.get("lang") or item.get("language") or "código"
    chosen = item.get("chosen") or item.get("safe_code") or ""
    rejected = item.get("rejected") or item.get("vulnerable_code") or ""
    question = item.get("question") or item.get("prompt") or item.get("instruction") or ""

    if not chosen and not rejected:
        return None

    # Formatar User Prompt
    if question and rejected and rejected not in question:
        user_prompt = f"{question.strip()}\n\n```{lang}\n{rejected.strip()}\n```"
    elif question:
        user_prompt = question.strip()
    elif rejected:
        user_prompt = (
            f"Analise o código {lang} abaixo em busca de vulnerabilidades de segurança ({vulnerability}). "
            f"Apresente seu raciocínio passo a passo identificando pontos de entrada, fluxo de dados inseguro, "
            f"sinks perigosos e forneça a remediação segura completa.\n\n"
            f"```{lang}\n{rejected.strip()}\n```"
        )
    else:
        user_prompt = f"Explique a mecânica de exploração e remediação segura para a seguinte vulnerabilidade: {vulnerability}."

    # Síntese determinística nativa dos 7 estágios analíticos CoT
    cot = (
        f"<think>\n"
        f"**Surface Analysis**: O componente sob auditoria é implementado em {lang}. "
        f"A lógica visa manipular dados e executar regras de negócio da aplicação.\n\n"
        f"**Entry Point / Attack Vector**: O ponto de entrada expõe a superfície de ataque a parâmetros "
        f"que podem ser controlados por fontes externas sem validação estrita de fronteira ou tipagem segura.\n\n"
        f"**Data Flow / Internal Mechanics (Taint Analysis)**: Os dados não confiáveis fluem através da "
        f"lógica do programa sem passar por filtros adequados ou sanitização antes de atingirem funções críticas.\n\n"
        f"**Vulnerable Sink / Control Flaw Detection**: O fluxo alcança um sink de execução ou consumo de memória inseguro, "
        f"revelando o padrão clássico de {vulnerability}.\n\n"
        f"**Sanitization and Mitigation Assessment**: Ausência de checagens defensivas, allowlists ou escaping "
        f"contextual na implementação examinada. A variante vulnerável falha em aplicar proteções em profundidade.\n\n"
        f"**Vulnerability Verdict / Anomaly Classification**: O código é classificado como VULNERÁVEL a {vulnerability}. "
        f"A falha compromete as garantias de segurança do módulo em tempo de execução.\n\n"
        f"**Potential Impact**: Comprometimento potencial de confidencialidade, integridade ou disponibilidade dos "
        f"recursos do sistema, dependendo dos privilégios de execução e do contexto de implantação.\n"
        f"</think>"
    )

    final_content = (
        f"FATO: A análise confirma a presença de vulnerabilidade classificada como {vulnerability}.\n\n"
        f"SUGESTÃO DE CORREÇÃO E IMPLEMENTAÇÃO SEGURA:\n\n{chosen.strip()}"
    )

    assistant_content = f"{cot}\n\n{final_content}"

    return {
        "messages": [
            {
                "role": "system",
                "content": "Você é o CyberRott, uma IA especialista em segurança ofensiva, defensiva e análise de vulnerabilidades com raciocínio analítico profundo."
            },
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content}
        ]
    }

# =============================================================================
# 5. EXTRATOR UNIVERSAL DE REGISTROS DO HUGGING FACE
# =============================================================================

def extract_qa_from_hf_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Inspeciona dinamicamente as colunas de um registro do Hugging Face e formata em ChatML."""
    # Caso 0: Registro de vulnerabilidade estruturado (CyberNative DPO)
    if ("chosen" in item or "rejected" in item) and ("vulnerability" in item or "cwe" in item or "lang" in item):
        return format_cybernative_dpo_item(item)

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

    # Caso 3: Campos de Pergunta / Problema
    question = (
        item.get("problem")
        or item.get("question")
        or item.get("prompt")
        or item.get("instruction")
        or item.get("query")
        or item.get("problem_description")
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
            or item.get("generation")
            or item.get("answer")
            or item.get("response")
            or item.get("output")
        )

    if question and solution_text:
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "Você é o CyberRott, uma IA especialista em segurança ofensiva, defensiva e análise de vulnerabilidades com raciocínio analítico profundo."
                },
                {"role": "user", "content": str(question).strip()},
                {"role": "assistant", "content": str(solution_text).strip()}
            ]
        }

    return None

# =============================================================================
# 6. DESDUPLICAÇÃO POR HASH SHA-256 DO PROMPT
# =============================================================================

def deduplicate_samples(
    samples: List[Dict[str, Any]],
    preserve_anchors: bool = True
) -> List[Dict[str, Any]]:
    """
    Remove amostras com prompts duplicados entre datasets externos,
    preservando o upsampling intencional das amostras âncora de ouro.
    """
    seen_prompts: Set[str] = set()
    deduped: List[Dict[str, Any]] = []
    duplicates_count = 0

    for item in samples:
        is_anchor = item.get("_is_anchor", False)

        # Extrair user prompt
        prompt = ""
        msgs = item.get("messages", [])
        for m in msgs:
            if m.get("role") == "user":
                prompt = m.get("content", "")
                break

        # Normalização do prompt para cálculo do hash
        normalized_prompt = re.sub(r"\s+", " ", prompt).strip().lower()
        prompt_hash = hashlib.sha256(normalized_prompt.encode("utf-8")).hexdigest()

        if is_anchor and preserve_anchors:
            clean_item = {k: v for k, v in item.items() if not k.startswith("_")}
            deduped.append(clean_item)
        elif prompt_hash in seen_prompts:
            duplicates_count += 1
            continue
        else:
            seen_prompts.add(prompt_hash)
            clean_item = {k: v for k, v in item.items() if not k.startswith("_")}
            deduped.append(clean_item)

    if duplicates_count > 0:
        print(f"[✓] Desduplicação por hash SHA-256: {duplicates_count} prompts duplicados removidos.")

    return deduped

# =============================================================================
# 7. CARREGAMENTO DAS AMOSTRAS ÂNCORA DE OURO
# =============================================================================

def load_anchor_samples(anchor_file: str) -> List[Dict[str, Any]]:
    """Carrega as amostras âncora no formato padronizado ChatML marcando _is_anchor=True."""
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
                    item = dict(data)
                    item["_is_anchor"] = True
                    samples.append(item)
                elif "instruction" in data and ("final_answer" in data or "reasoning_trace" in data):
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
                        "_is_anchor": True,
                        "messages": [
                            {"role": "system", "content": "Você é o CyberRott, uma IA especialista em segurança ofensiva, defensiva e análise de vulnerabilidades com raciocínio analítico profundo."},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": assistant_content}
                        ]
                    })
            except Exception as e:
                print(f"[Aviso] Falha ao processar linha {idx} do arquivo âncora: {e}")
                continue

    return samples

# =============================================================================
# 8. STREAMING E INGESTÃO DO HUGGING FACE
# =============================================================================

def fetch_single_dataset_stream(
    dataset_name: str,
    max_samples: int,
    cyber_regex: Optional[re.Pattern] = None,
    candidate_splits: Optional[List[str]] = None,
    scrub_refusals: bool = True,
    enforce_think: bool = True
) -> List[Dict[str, Any]]:
    """Carrega via streaming um dataset do Hugging Face aplicando filtragem e padronização."""
    collected = []
    print(f"[*] Conectando ao Hugging Face para streaming do dataset '{dataset_name}'...")

    try:
        from datasets import load_dataset
    except ImportError:
        print("[Aviso] Biblioteca 'datasets' não instalada. Execute 'pip install datasets'.")
        return []

    splits = candidate_splits or ["train", "code", "stem", "math"]
    ds = None
    for split_choice in splits:
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
            print(f"[Aviso] Não foi possível inicializar streaming de '{dataset_name}': {e}")
            return []

    count = 0
    checked = 0
    try:
        for item in ds:
            checked += 1

            # Filtragem de segurança se exigida
            if cyber_regex is not None:
                text_parts = []
                for v in item.values():
                    if isinstance(v, (str, int, float)):
                        text_parts.append(str(v))
                    elif isinstance(v, list):
                        for sub in v:
                            if isinstance(sub, str):
                                text_parts.append(sub)
                            elif isinstance(sub, dict):
                                text_parts.extend(str(val) for val in sub.values() if isinstance(val, (str, int, float)))
                full_text = " ".join(text_parts)
                if not cyber_regex.search(full_text):
                    continue

            # Formatação para ChatML
            formatted = extract_qa_from_hf_item(item)
            if not formatted:
                continue

            # Scrubber de recusas e disclaimers
            if scrub_refusals:
                scrubbed = scrub_refusal_and_boilerplate(formatted)
                if not scrubbed:
                    continue
                formatted = scrubbed

            # Enforcing de tags <think>
            if enforce_think:
                msgs = formatted.get("messages", [])
                for m in msgs:
                    if m.get("role") == "assistant":
                        m["content"] = format_to_cyberrott_cot(m.get("content", ""), enforce_think=True)

            collected.append(formatted)
            count += 1
            if count % 100 == 0:
                print(f"[*] Ingestão '{dataset_name}': {count}/{max_samples} amostras (analisadas {checked})...")
            if count >= max_samples:
                break
    except Exception as e:
        print(f"[Aviso] Interrupção durante o streaming de '{dataset_name}': {e}")

    print(f"[✓] Concluído '{dataset_name}': {len(collected)} amostras coletadas com sucesso.")
    return collected

def fetch_hf_cyber_samples(
    dataset_name: str,
    max_samples: int,
    cyber_regex: re.Pattern,
    scrub_refusals: bool = False,
    enforce_think: bool = False
) -> List[Dict[str, Any]]:
    """Função legada mantida para retrocompatibilidade total com a v1.0."""
    return fetch_single_dataset_stream(
        dataset_name=dataset_name,
        max_samples=max_samples,
        cyber_regex=cyber_regex,
        candidate_splits=["train", "code", "stem", "math"],
        scrub_refusals=scrub_refusals,
        enforce_think=enforce_think
    )

def fetch_multi_hf_datasets(
    dataset_keys: List[str],
    cyber_regex: re.Pattern,
    custom_limits: Optional[Dict[str, int]] = None,
    scrub_refusals: bool = True,
    enforce_think: bool = True
) -> List[Dict[str, Any]]:
    """Coleta e agrega amostras de múltiplos datasets do catálogo v1.1."""
    all_hf_samples = []

    for key in dataset_keys:
        if key not in CYBERROTT_V1_1_DATASETS:
            print(f"[Aviso] Dataset '{key}' desconhecido no catálogo. Ignorando.")
            continue

        cfg = CYBERROTT_V1_1_DATASETS[key]
        target = custom_limits.get(key, cfg["target_samples"]) if custom_limits else cfg["target_samples"]
        repo = cfg["repo_id"]
        use_regex = cyber_regex if cfg["requires_cyber_filter"] else None

        print(f"\n---> Coletando [{cfg['name']}] (Alvo: {target} amostras | Domínio: {cfg['domain']})")

        samples = fetch_single_dataset_stream(
            dataset_name=repo,
            max_samples=target,
            cyber_regex=use_regex,
            candidate_splits=cfg.get("splits"),
            scrub_refusals=scrub_refusals,
            enforce_think=enforce_think
        )

        # Fallback se aplicável e se a coleta do repo principal falhou totalmente
        if not samples and "fallback_repo_id" in cfg:
            fallback_repo = cfg["fallback_repo_id"]
            print(f"[!] Tentando repositório de fallback: '{fallback_repo}'...")
            samples = fetch_single_dataset_stream(
                dataset_name=fallback_repo,
                max_samples=target,
                cyber_regex=use_regex,
                candidate_splits=cfg.get("splits"),
                scrub_refusals=scrub_refusals,
                enforce_think=enforce_think
            )

        all_hf_samples.extend(samples)

    return all_hf_samples

# =============================================================================
# 9. EXIBIÇÃO DO CATÁLOGO DE DATASETS CLI (--list_datasets)
# =============================================================================

def print_dataset_catalog():
    """Imprime a tabela descritiva do catálogo v1.1 de datasets open-source."""
    total_target = sum(cfg["target_samples"] for cfg in CYBERROTT_V1_1_DATASETS.values())
    header = "=" * 105
    print("\n" + header)
    print("🛡️  CYBERROTT 1.1 - CATÁLOGO OFICIAL DE DATASETS ZERO-API COST (ROADMAP v1.1)")
    print(header)
    print(f"{'CHAVE':<16} {'HF REPOSITORY':<44} {'ALVO (v1.1)':<12} {'FILTRO CYBER':<14} {'DOMÍNIO TÉCNICO'}")
    print("-" * 105)
    for key, cfg in CYBERROTT_V1_1_DATASETS.items():
        filter_str = "Sim" if cfg["requires_cyber_filter"] else "Não (Nativo)"
        print(f"{key:<16} {cfg['repo_id']:<44} {cfg['target_samples']:<12} {filter_str:<14} {cfg['domain']}")
    print("-" * 105)
    print(f"TOTAL ALVO EXTERNO: ~{total_target:,} sequências de alta densidade analítica.")
    print(f"TOTAL COM ÂNCORAS DE OURO: ~{total_target + 320:,} sequências (cumprindo a meta de ~10,000 do Roadmap v1.1).")
    print(header + "\n")

# =============================================================================
# 10. PIPELINE PRINCIPAL DE FUSÃO
# =============================================================================

def run_fusion_pipeline(
    anchor_file: str = "dataset_cyber_ready.jsonl",
    anchor_repeat: int = 20,
    preset: str = "v1.0",
    dataset_keys: Optional[List[str]] = None,
    hf_dataset: Optional[str] = None,
    max_hf_samples: int = 2500,
    scrub_refusals: bool = True,
    enforce_think: bool = True,
    output_file: str = "cyberrott_final_train.jsonl",
    seed: int = 42
):
    """
    Executa o pipeline unificado de preparação:
    - Anchor Mixing com upsampling de exemplos de ouro.
    - Streaming do Hugging Face (modo v1.0 clássico ou v1.1 multi-dataset expandido).
    - Scrubber de recusas e disclaimers.
    - Enforcing de tags <think> CoT.
    - Desduplicação por hash SHA-256 e shuffling determinístico.
    """
    print("=" * 75)
    print(f"🛡️  CyberRott Pipeline de Fusão de Dados de Treinamento (Preset: {preset.upper()})")
    print("=" * 75)
    print(f"• Arquivo Âncora: {anchor_file} (Upsampling: {anchor_repeat}x)")
    print(f"• Saída Final:    {output_file}")
    print(f"• Random Seed:    {seed}")
    print(f"• Scrub Refusals: {'Ativado' if scrub_refusals else 'Desativado'}")
    print(f"• Enforce <think>: {'Ativado' if enforce_think else 'Desativado'}")
    print("-" * 75)

    # 1. Carregar Amostras Âncora
    raw_anchors = load_anchor_samples(anchor_file)
    unique_anchor_count = len(raw_anchors)
    anchor_samples = raw_anchors * anchor_repeat
    print(f"[✓] Âncoras carregadas: {unique_anchor_count} únicas -> {len(anchor_samples)} após upsampling ({anchor_repeat}x).")

    # 2. Ingestão de Datasets do Hugging Face
    cyber_regex = compile_cyber_regex(CYBER_KEYWORDS)
    hf_samples = []

    if preset == "v1.1":
        active_keys = dataset_keys or list(CYBERROTT_V1_1_DATASETS.keys())
        print(f"[*] Modo v1.1 Ativado: Processando {len(active_keys)} datasets do catálogo...")
        hf_samples = fetch_multi_hf_datasets(
            dataset_keys=active_keys,
            cyber_regex=cyber_regex,
            scrub_refusals=scrub_refusals,
            enforce_think=enforce_think
        )
    else:
        # Modo clássico v1.0 (retrocompatibilidade)
        target_repo = hf_dataset or "nvidia/Nemotron-Research-GooseReason-0.7M"
        if max_hf_samples > 0:
            print(f"[*] Modo v1.0 Clássico: Ingestão de '{target_repo}' (máx: {max_hf_samples})...")
            try:
                hf_samples = fetch_single_dataset_stream(
                    dataset_name=target_repo,
                    max_samples=max_hf_samples,
                    cyber_regex=cyber_regex,
                    scrub_refusals=scrub_refusals,
                    enforce_think=enforce_think
                )
            except Exception as e:
                print(f"[Aviso] Falha de conexão ou erro no Hugging Face: {e}")

    # 3. Combinar e Desduplicar
    combined = anchor_samples + hf_samples
    if not combined:
        print("[Erro Crítico] Nenhuma amostra disponível para compor o dataset de treinamento!")
        return

    deduped_samples = deduplicate_samples(combined, preserve_anchors=True)

    # 4. Embaralhamento Determinístico
    random.seed(seed)
    random.shuffle(deduped_samples)

    # 5. Gravação em Disco
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f_out:
        for entry in deduped_samples:
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 6. Relatório Final Consolidado
    print("\n" + "=" * 75)
    print("📊 RELATÓRIO CONSOLIDADO DO DATASET DE TREINAMENTO")
    print("=" * 75)
    print(f"• Preset Executado:                         {preset.upper()}")
    print(f"• Total de Âncoras de Ouro (únicas):        {unique_anchor_count}")
    print(f"• Fator de Upsampling das Âncoras:          {anchor_repeat}x")
    print(f"• Total de Amostras Âncora no Treino:       {len(anchor_samples)}")
    print(f"• Total de Amostras Hugging Face Coletadas: {len(hf_samples)}")
    print(f"• TOTAL FINAL DESDUPLICADO E EMBARALHADO:   {len(deduped_samples)}")
    print(f"• Arquivo Gerado:                           {output_file}")
    print("=" * 75 + "\n")

# =============================================================================
# 11. INTERFACE DE LINHA DE COMANDO (CLI)
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de Fusão e Preparação de Dados para o CyberRott (v1.0 & v1.1)"
    )

    # Controle de Versão / Preset
    parser.add_argument(
        "--preset",
        choices=["v1.0", "v1.1"],
        default="v1.0",
        help="Preset de dados: 'v1.0' (GooseReason clássico 2.5k) ou 'v1.1' (Multi-Dataset ~10k CoT)"
    )
    parser.add_argument(
        "--list_datasets",
        action="store_true",
        help="Exibe o catálogo oficial de datasets do Roadmap v1.1 e encerra a execução"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Lista customizada de chaves do catálogo para ingestão (ex: goose_reason code_feedback cyber_native)"
    )

    # Parâmetros Legados e de Configuração
    parser.add_argument(
        "--anchor_file",
        default="dataset_cyber_ready.jsonl",
        help="Caminho do arquivo com os 16 exemplos complexos de ouro"
    )
    parser.add_argument(
        "--anchor_repeat",
        type=int,
        default=20,
        help="Fator de repetição (upsampling) para as amostras âncora (default: 20 -> 320 amostras)"
    )
    parser.add_argument(
        "--hf_dataset",
        default="nvidia/Nemotron-Research-GooseReason-0.7M",
        help="Dataset no Hugging Face para streaming (usado no modo v1.0 ou override)"
    )
    parser.add_argument(
        "--max_hf_samples",
        type=int,
        default=2500,
        help="Quantidade máxima de amostras a extrair do Hugging Face no modo v1.0"
    )
    parser.add_argument(
        "--output",
        default="cyberrott_final_train.jsonl",
        help="Caminho do arquivo JSONL de saída final"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reproduzibilidade do embaralhamento (shuffling)"
    )

    # Flags de Higienização e CoT
    parser.add_argument(
        "--scrub_refusals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ativar/desativar scrubber de recusas conversacionais e disclaimers moralistas"
    )
    parser.add_argument(
        "--enforce_think_tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Garantir e estruturar blocos <think>...</think> de raciocínio CoT em todas as respostas"
    )

    args = parser.parse_args()

    if args.list_datasets:
        print_dataset_catalog()
        sys.exit(0)

    run_fusion_pipeline(
        anchor_file=args.anchor_file,
        anchor_repeat=args.anchor_repeat,
        preset=args.preset,
        dataset_keys=args.datasets,
        hf_dataset=args.hf_dataset,
        max_hf_samples=args.max_hf_samples,
        scrub_refusals=args.scrub_refusals,
        enforce_think=args.enforce_think_tags,
        output_file=args.output,
        seed=args.seed
    )
