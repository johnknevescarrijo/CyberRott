#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CyberRott 1.1 - Unified Data Preparation and Fusion Pipeline
=============================================================
(Pillar 1: Zero-API Cost Data Expansion - 100% Open-Source & Community Tiers).

Functionalities:
1. Anchor Mixing: Complex Golden Anchors with weighted upsampling (20x).
2. Official v1.1 Catalog: Streaming multi-dataset ingestion from Hugging Face Hub:
   - nvidia/Nemotron-Research-GooseReason-0.7M (Deep reasoning & code logic)
   - m-a-p/CodeFeedback-Filtered-Instruction (Code audit, memory vulnerabilities & refactoring)
   - open-r1/codeforces-cots / bespokelabs/Bespoke-Stratos-17k (Long-CoT trajectories)
   - CyberNative/Code_Vulnerability_Security_DPO (Vulnerability detection & DPO patches)
   - AlicanKiraz0/Cybersecurity-Dataset-v1 (Red/Blue Team, MITRE ATT&CK & forensics)
3. Programmatic Alignment Refusal Scrubber and Moralizing Disclaimer Cleaner.
4. Automated 7-Stage Native CoT Formatter (<think>...</think>).
5. Deterministic SHA-256 Prompt Deduplication.
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
# 1. OFFICIAL OPEN-SOURCE DATASET CATALOG (ZERO-API COST)
# =============================================================================

CYBERROTT_V1_1_DATASETS: Dict[str, Dict[str, Any]] = {
    "goose_reason": {
        "repo_id": "nvidia/Nemotron-Research-GooseReason-0.7M",
        "name": "NVIDIA GooseReason-0.7M",
        "target_samples": 1500,
        "requires_cyber_filter": True,
        "domain": "Deep Reasoning & Code Logic",
        "description": "Primary analytical reasoning corpus from NVIDIA featuring formal multi-step deduction chains.",
        "splits": ["train", "code", "stem", "math"],
    },
    "code_feedback": {
        "repo_id": "m-a-p/CodeFeedback-Filtered-Instruction",
        "name": "CodeFeedback Filtered Instruction",
        "target_samples": 1000,
        "requires_cyber_filter": True,
        "domain": "Code Audit, Memory & Refactoring",
        "description": "Source code review trajectories, memory corruption fixes, and defensive refactoring.",
        "splits": ["train"],
    },
    "open_r1": {
        "repo_id": "open-r1/codeforces-cots",
        "fallback_repo_id": "bespokelabs/Bespoke-Stratos-17k",
        "name": "Open-R1 CoTs (Fallback: Bespoke-Stratos)",
        "target_samples": 800,
        "requires_cyber_filter": True,
        "domain": "Long-CoT Reasoning & Trajectories",
        "description": "Open community long Chain-of-Thought reasoning traces with structured reflections.",
        "splits": ["train"],
    },
    "cyber_native": {
        "repo_id": "CyberNative/Code_Vulnerability_Security_DPO",
        "name": "CyberNative Vulnerability Security DPO",
        "target_samples": 500,
        "requires_cyber_filter": False,
        "domain": "Code Vulnerabilities & CWE DPO Patches",
        "description": "Pairwise vulnerable code (rejected) vs. secure implementation (chosen) mapped to CWE taxonomy.",
        "splits": ["train"],
    },
    "cybersecurity_v1": {
        "repo_id": "AlicanKiraz0/Cybersecurity-Dataset-v1",
        "name": "AlicanKiraz0 Cybersecurity Dataset v1",
        "target_samples": 400,
        "requires_cyber_filter": False,
        "domain": "Red/Blue Team, MITRE ATT&CK & Forensics",
        "description": "Domain-specialized dataset covering offensive tactics, defense evasion, CVEs, and digital forensics.",
        "splits": ["train"],
    },
}

# =============================================================================
# 2. IN-DEPTH CYBERSECURITY & SYSTEM LEVEL KEYWORDS LIST
# =============================================================================

CYBER_KEYWORDS: List[str] = [
    # Vulnerabilities and Exploitation
    "vulnerability", "exploit", "cve", "cve-", "buffer overflow", "stack overflow",
    "heap overflow", "use-after-free", "out-of-bounds", "format string",
    "race condition", "toctou", "pwn", "shellcode", "rop chain", "ret2libc",
    "privilege escalation", "bypass", "taint analysis", "injection",
    "sql injection", "sqli", "cross-site scripting", "xss", "csrf", "ssrf",
    "command injection", "path traversal", "directory traversal",
    # Reverse Engineering and Binaries
    "reverse engineering", "disassembly", "decompiler", "decompilation",
    "gdb", "radare2", "ghidra", "ida pro", "binwalk", "objdump",
    "elf", "pe binary", "assembly", "x86", "x86_64", "arm64",
    # Malware and Forensics
    "malware", "ransomware", "backdoor", "trojan", "rootkit", "keylogger",
    "botnet", "forensics", "memory dump", "volatility", "sysmon",
    "wireshark", "pcap", "packet analysis", "network traffic", "yara", "sigma",
    # Networks and Protocols
    "tcp/ip", "packet", "dns", "dns tunneling", "socket", "http request",
    "firewall", "ids", "ips", "edr", "mitre", "port scan", "nmap",
    # Cryptography and Authentication
    "cryptography", "rsa", "aes", "elliptic curve", "diffie-hellman",
    "sha256", "jwt", "hash collision", "nonce reuse", "kerberos",
    "active directory", "golden ticket", "silver ticket", "pass-the-hash",
    "authentication", "authorization", "sandbox", "kernel exploit", "container escape"
]


def compile_cyber_regex(keywords: List[str]) -> re.Pattern:
    """Compiles regex with word-boundaries for short terms to eliminate false positives."""
    patterns = []
    for kw in keywords:
        if len(kw) <= 4:
            patterns.append(rf"\b{re.escape(kw)}\b")
        else:
            patterns.append(re.escape(kw))
    return re.compile("|".join(patterns), re.IGNORECASE)


# =============================================================================
# 3. REFUSAL SCRUBBER AND MORALIZING DISCLAIMER CLEANER
# =============================================================================

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

DISCLAIMER_PATTERNS: List[str] = [
    r"(?im)^(?:disclaimer|note|warning|aviso|atenção|important notice)[\s\:\-]+.*?(?:educational purposes?|authorized (?:testing|systems?|environments?)|without (?:prior )?(?:written )?permission|illegal|fins educacionais).*?\n+",
    r"(?im)\n+(?:disclaimer|note|warning|aviso)[\s\:\-]+.*?(?:educational purposes?|authorized (?:testing|systems?|environments?)|without (?:prior )?(?:written )?permission|illegal|fins educacionais).*?$",
    r"(?im)\b(?:please note that|remember that|keep in mind that)\s+(?:hacking without permission|unauthorized access to computer systems|attacking systems)\s+is illegal.*?\n*",
    r"(?im)\bthis information is strictly for educational and defensive purposes only\.?\b",
    r"(?im)\balways ensure you have explicit (?:written )?permission before (?:testing|accessing).*?\b",
    r"(?im)\bdisclaimer:\s*i cannot condone malicious activities\.?\b",
]


def is_conversational_refusal(text: str) -> bool:
    """Verifies whether the response substantially consists of alignment refusals."""
    if not text:
        return True

    clean_text = text.strip()

    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, clean_text, re.IGNORECASE):
            # Short response (< 600 chars) with refusal marker: reject
            if len(clean_text) < 600:
                return True
            # Starts with refusal statement and lacks formatted code block: reject
            if re.search(r"^(?:i cannot|as an ai|i'm sorry|i apologize)", clean_text, re.IGNORECASE) and "```" not in clean_text:
                return True
            # Contains only a few refusal declaratory lines: reject
            lines = [l.strip() for l in clean_text.split("\n") if l.strip()]
            if len(lines) <= 4:
                return True

    return False


def clean_moralizing_disclaimers(text: str) -> str:
    """Removes moralizing disclaimers and empty preaching boilerplate at start/end of responses."""
    if not text:
        return text

    cleaned = text
    for pattern in DISCLAIMER_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned).strip()

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def scrub_refusal_and_boilerplate(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Applies the full scrubber pipeline to a ChatML sample.
    Returns cleaned sample or None if the assistant response is an unrecoverable refusal.
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
# 4. 7-STAGE CoT REASONING FORMATTER (<think>...</think>)
# =============================================================================

def format_to_cyberrott_cot(text: str, enforce_think: bool = True) -> str:
    """
    Normalizes alternative reasoning tags (<|begin_of_thought|>, <thought>, <reasoning>)
    into canonical <think>...</think> tags and formats unstructured responses into CoT blocks.
    """
    if not text:
        return text

    # 1. Normalize alternative tags
    normalized = text
    normalized = re.sub(r"<\s*\|begin_of_thought\|\s*>", "<think>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"<\s*\|end_of_thought\|\s*>", "</think>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"<\s*thought\s*>", "<think>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"<\s*/\s*thought\s*>", "</think>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"<\s*reasoning\s*>", "<think>", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"<\s*/\s*reasoning\s*>", "</think>", normalized, flags=re.IGNORECASE)

    # 2. If balanced tags exist, return directly
    if "<think>" in normalized and "</think>" in normalized:
        return normalized

    # 3. If open tag exists without closing tag, close it
    if "<think>" in normalized and "</think>" not in normalized:
        return normalized + "\n</think>"

    # 4. If no think tag and enforcement is active
    if enforce_think:
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

        paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
        if len(paragraphs) >= 2:
            reasoning = paragraphs[0]
            conclusion = "\n\n".join(paragraphs[1:])
            return f"<think>\n**Technical Analysis**: {reasoning}\n</think>\n\n{conclusion}"
        else:
            return f"<think>\n**Preliminary Analysis**: Assessment of parameters and problem mechanics.\n</think>\n\n{normalized}"

    return normalized


def format_cybernative_dpo_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Deterministically synthesizes an audit sample in the canonical 7-stage format
    from DPO vulnerability metadata (chosen vs rejected).
    """
    vulnerability = (
        item.get("vulnerability")
        or item.get("vuln_type")
        or item.get("cwe")
        or "Code Security Vulnerability"
    )
    lang = item.get("lang") or item.get("language") or "code"
    chosen = item.get("chosen") or item.get("safe_code") or ""
    rejected = item.get("rejected") or item.get("vulnerable_code") or ""
    question = item.get("question") or item.get("prompt") or item.get("instruction") or ""

    if not chosen and not rejected:
        return None

    if question and rejected and rejected not in question:
        user_prompt = f"{question.strip()}\n\n```{lang}\n{rejected.strip()}\n```"
    elif question:
        user_prompt = question.strip()
    elif rejected:
        user_prompt = (
            f"Analyze the {lang} code below for security vulnerabilities ({vulnerability}). "
            f"Provide your step-by-step reasoning identifying entry points, insecure data flow, "
            f"dangerous sinks, and furnish the complete secure remediation.\n\n"
            f"```{lang}\n{rejected.strip()}\n```"
        )
    else:
        user_prompt = f"Explain the exploitation mechanics and secure remediation for: {vulnerability}."

    # Native synthesis of 7 analytical CoT stages (tested in test_data_prep_v1_1.py)
    cot = (
        f"<think>\n"
        f"**Surface Analysis**: The target component is implemented in {lang}. "
        f"The logic operates on inputs and enforces application business constraints.\n\n"
        f"**Entry Point / Attack Vector**: The entry point exposes attack surface to parameters "
        f"controlled by external untrusted sources without strict boundary checks.\n\n"
        f"**Data Flow / Internal Mechanics**: Tainted input propagates across internal routines "
        f"without adequate intermediate sanitization checkpoints.\n\n"
        f"**Vulnerable Sink / Control Flaw Detection**: Execution reaches a dangerous sink, "
        f"manifesting {vulnerability}.\n\n"
        f"**Sanitization and Mitigation Assessment**: Defensive validation allowlists and contextual "
        f"escaping are absent in the vulnerable implementation.\n\n"
        f"**Vulnerability Verdict**: The implementation is confirmed VULNERABLE to {vulnerability}.\n\n"
        f"**Potential Impact**: Risk of unauthorized confidentiality loss, integrity corruption, or resource exhaustion.\n"
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
                "content": "You are CyberRott, a specialized reasoning AI for offensive, defensive, and vulnerability analysis."
            },
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content}
        ]
    }


# =============================================================================
# 5. UNIVERSAL HUGGING FACE RECORD EXTRACTOR
# =============================================================================

def extract_qa_from_hf_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Dynamically inspects columns of a Hugging Face record and formats into ChatML schema."""
    if ("chosen" in item or "rejected" in item) and ("vulnerability" in item or "cwe" in item or "lang" in item):
        return format_cybernative_dpo_item(item)

    if "messages" in item and isinstance(item["messages"], list):
        return {"messages": item["messages"]}

    if "conversations" in item and isinstance(item["conversations"], list):
        role_map = {"human": "user", "gpt": "assistant", "chatgpt": "assistant", "system": "system"}
        msgs = []
        for c in item["conversations"]:
            role = role_map.get(c.get("from", "").lower(), "user")
            msgs.append({"role": role, "content": str(c.get("value", ""))})
        if msgs:
            return {"messages": msgs}

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

    options = item.get("options")
    answer_raw = item.get("answer")
    solution_text = None

    if options and isinstance(options, list) and answer_raw:
        ans_str = str(answer_raw).strip().upper()
        if len(ans_str) == 1 and "A" <= ans_str <= "Z":
            idx = ord(ans_str) - ord("A")
            if 0 <= idx < len(options):
                solution_text = f"Correct choice: **{ans_str}**\n\n```\n{options[idx]}\n```"
            else:
                solution_text = f"Correct choice: \\boxed{{{ans_str}}}"

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
                    "content": "You are CyberRott, a specialized reasoning AI for offensive, defensive, and vulnerability analysis."
                },
                {"role": "user", "content": str(question).strip()},
                {"role": "assistant", "content": str(solution_text).strip()}
            ]
        }

    return None


# =============================================================================
# 6. SHA-256 PROMPT DEDUPLICATION
# =============================================================================

def deduplicate_samples(
    samples: List[Dict[str, Any]],
    preserve_anchors: bool = True
) -> List[Dict[str, Any]]:
    """
    Removes samples with duplicated prompts across external datasets,
    preserving intentional upsampling for Golden Anchor samples.
    """
    seen_prompts: Set[str] = set()
    deduped: List[Dict[str, Any]] = []
    duplicates_count = 0

    for item in samples:
        is_anchor = item.get("_is_anchor", False)

        prompt = ""
        msgs = item.get("messages", [])
        for m in msgs:
            if m.get("role") == "user":
                prompt = m.get("content", "")
                break

        normalized_prompt = re.sub(r"\s+", " ", prompt).strip().lower()
        prompt_hash = hashlib.sha256(normalized_prompt.encode("utf-8")).hexdigest()

        if is_anchor and preserve_anchors:
            seen_prompts.add(prompt_hash)
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
        print(f"[✓] SHA-256 prompt deduplication: {duplicates_count} duplicate prompts removed.")

    return deduped


# =============================================================================
# 7. LOADING GOLDEN ANCHOR SAMPLES
# =============================================================================

def load_anchor_samples(anchor_file: str) -> List[Dict[str, Any]]:
    """Loads Golden Anchor samples in ChatML schema, tagging them with _is_anchor=True."""
    if not os.path.exists(anchor_file):
        fallback = os.path.join("data", anchor_file)
        if os.path.exists(fallback):
            print(f"[*] Anchor fallback: '{anchor_file}' -> '{fallback}'")
            anchor_file = fallback
        else:
            print(f"[Warning] Anchor file '{anchor_file}' not found.")
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
                            {"role": "system", "content": "You are CyberRott, a specialized reasoning AI for offensive, defensive, and vulnerability analysis."},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": assistant_content}
                        ]
                    })
            except Exception as e:
                print(f"[Warning] Failed to process line {idx} in anchor file: {e}")
                continue

    return samples


# =============================================================================
# 8. HUGGING FACE STREAMING & INGESTION
# =============================================================================

def fetch_single_dataset_stream(
    dataset_name: str,
    max_samples: int,
    cyber_regex: Optional[re.Pattern] = None,
    candidate_splits: Optional[List[str]] = None,
    scrub_refusals: bool = True,
    enforce_think: bool = True
) -> List[Dict[str, Any]]:
    """Streams a Hugging Face dataset applying domain filtering, refusal scrubbing, and CoT formatting."""
    collected = []
    print(f"[*] Connecting to Hugging Face Hub to stream '{dataset_name}'...")

    try:
        from datasets import load_dataset
    except ImportError:
        print("[Warning] 'datasets' library is not installed. Run 'pip install datasets'.")
        return []

    splits = candidate_splits or ["train", "code", "stem", "math"]
    ds = None
    for split_choice in splits:
        try:
            ds = load_dataset(dataset_name, split=split_choice, streaming=True)
            print(f"[✓] Streaming established via split='{split_choice}'.")
            break
        except Exception:
            continue

    if ds is None:
        try:
            ds_dict = load_dataset(dataset_name, streaming=True)
            active_split = "code" if "code" in ds_dict else list(ds_dict.keys())[0]
            ds = ds_dict[active_split]
            print(f"[✓] Streaming established via dynamic split='{active_split}'.")
        except Exception as e:
            print(f"[Warning] Unable to initialize streaming for '{dataset_name}': {e}")
            return []

    count = 0
    checked = 0
    max_checked_items = max_samples * 25
    try:
        for item in ds:
            checked += 1
            if checked > max_checked_items:
                print(f"[!] Maximum inspection threshold ({max_checked_items:,}) reached for '{dataset_name}'. Concluding with {count} samples.")
                break

            # Domain keyword filtering
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

            # Format to ChatML
            formatted = extract_qa_from_hf_item(item)
            if not formatted:
                continue

            # Scrub refusals and moralizing boilerplate
            if scrub_refusals:
                scrubbed = scrub_refusal_and_boilerplate(formatted)
                if not scrubbed:
                    continue
                formatted = scrubbed

            # Enforce <think> deliberation tags
            if enforce_think:
                msgs = formatted.get("messages", [])
                for m in msgs:
                    if m.get("role") == "assistant":
                        m["content"] = format_to_cyberrott_cot(m.get("content", ""), enforce_think=True)

            collected.append(formatted)
            count += 1
            if count % 100 == 0:
                print(f"[*] Ingestion '{dataset_name}': {count}/{max_samples} samples (inspected {checked})...")
            if count >= max_samples:
                break
    except Exception as e:
        print(f"[Warning] Interruption during streaming of '{dataset_name}': {e}")

    print(f"[✓] Completed '{dataset_name}': {len(collected)} samples successfully ingested.")
    return collected


def fetch_hf_cyber_samples(
    dataset_name: str,
    max_samples: int,
    cyber_regex: re.Pattern,
    scrub_refusals: bool = False,
    enforce_think: bool = False
) -> List[Dict[str, Any]]:
    """Legacy function maintained for backwards compatibility with v1.0."""
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
    """Aggregates samples from multiple datasets in the official catalog."""
    all_hf_samples = []

    for key in dataset_keys:
        if key not in CYBERROTT_V1_1_DATASETS:
            print(f"[Warning] Dataset key '{key}' unknown in catalog. Skipping.")
            continue

        cfg = CYBERROTT_V1_1_DATASETS[key]
        target = custom_limits.get(key, cfg["target_samples"]) if custom_limits else cfg["target_samples"]
        repo = cfg["repo_id"]
        use_regex = cyber_regex if cfg["requires_cyber_filter"] else None

        print(f"\n---> Ingesting [{cfg['name']}] (Target: {target} samples | Domain: {cfg['domain']})")

        samples = fetch_single_dataset_stream(
            dataset_name=repo,
            max_samples=target,
            cyber_regex=use_regex,
            candidate_splits=cfg.get("splits"),
            scrub_refusals=scrub_refusals,
            enforce_think=enforce_think
        )

        if not samples and "fallback_repo_id" in cfg:
            fallback_repo = cfg["fallback_repo_id"]
            print(f"[!] Attempting fallback repository: '{fallback_repo}'...")
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
# 9. DATASET CATALOG DISPLAY CLI (--list_datasets)
# =============================================================================

def print_dataset_catalog():
    """Prints descriptive table of the official Zero-API Cost dataset catalog."""
    total_target = sum(cfg["target_samples"] for cfg in CYBERROTT_V1_1_DATASETS.values())
    header = "=" * 105
    print("\n" + header)
    print("🛡️  CYBERROTT 1.1 - CATÁLOGO OFICIAL DE DATASETS ZERO-API COST (OFFICIAL DATASET CATALOG)")
    print(header)
    print(f"{'KEY':<16} {'HF REPOSITORY':<44} {'TARGET (v1.1)':<14} {'CYBER FILTER':<14} {'TECHNICAL DOMAIN'}")
    print("-" * 105)
    for key, cfg in CYBERROTT_V1_1_DATASETS.items():
        filter_str = "Yes" if cfg["requires_cyber_filter"] else "No (Native)"
        print(f"{key:<16} {cfg['repo_id']:<44} {cfg['target_samples']:<14} {filter_str:<14} {cfg['domain']}")
    print("-" * 105)
    print(f"TOTAL ALVO EXTERNO: ~{total_target:,} high-density analytical sequences.")
    print(f"TOTAL WITH GOLD ANCHORS: ~{total_target + 320:,} sequences (balanced high-density dataset).")
    print(header + "\n")


# =============================================================================
# 10. MAIN PREPARATION AND FUSION PIPELINE (prepare_dataset)
# =============================================================================

def prepare_dataset(
    anchor_file: str = "dataset_cyber_ready.jsonl",
    anchor_repeat: int = 20,
    preset: str = "v1.0",
    dataset_keys: Optional[List[str]] = None,
    hf_dataset: Optional[str] = None,
    max_hf_samples: int = 2500,
    scrub_refusals: bool = True,
    enforce_think: bool = True,
    output_file: str = "cyberrott_final_train.jsonl",
    seed: int = 42,
    goose_limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Main CyberRott training dataset preparation and fusion pipeline:
    - Automatic fallback if anchor_file is not in root path (searches in 'data/').
    - Catalog multi-dataset streaming (v1.1) or single-dataset streaming (v1.0).
    - Refusal scrubbing and moralizing disclaimer removal.
    - Tag normalization into canonical <think>...</think> schema.
    - Deterministic deduplication with Golden Anchor preservation.
    - Shuffling and export to target JSONL file.
    """
    if goose_limit is not None:
        CYBERROTT_V1_1_DATASETS["goose_reason"]["target_samples"] = goose_limit

    if not os.path.exists(anchor_file):
        fallback_candidate = os.path.join("data", anchor_file)
        if os.path.exists(fallback_candidate):
            print(f"[*] Anchor fallback applied in prepare_dataset: '{anchor_file}' -> '{fallback_candidate}'")
            anchor_file = fallback_candidate

    print("=" * 75)
    print(f"🛡️  CyberRott Training Data Fusion Pipeline (Preset: {preset.upper()})")
    print("=" * 75)
    print(f"• Anchor File:    {anchor_file} (Upsampling: {anchor_repeat}x)")
    print(f"• Target Output:  {output_file}")
    print(f"• Random Seed:    {seed}")
    print(f"• Scrub Refusals: {'Enabled' if scrub_refusals else 'Disabled'}")
    print(f"• Enforce <think>:{'Enabled' if enforce_think else 'Disabled'}")
    if goose_limit is not None:
        print(f"• Goose Limit:    {goose_limit}")
    print("-" * 75)

    # 1. Load Anchor Samples
    raw_anchors = load_anchor_samples(anchor_file)
    unique_anchor_count = len(raw_anchors)
    anchor_samples = raw_anchors * anchor_repeat
    print(f"[✓] Anchors loaded: {unique_anchor_count} unique -> {len(anchor_samples)} after upsampling ({anchor_repeat}x).")

    # 2. Local Base Reuse
    local_base_samples: List[Dict[str, Any]] = []
    reusing_local_v1 = False

    if preset == "v1.1" and dataset_keys is None and goose_limit is None:
        candidate_base = None
        if os.path.exists(output_file):
            candidate_base = output_file
        elif output_file in ["cyberrott_final_train.jsonl", "data/cyberrott_final_train.jsonl"]:
            for cand in ["data/cyberrott_final_train.jsonl", "cyberrott_final_train.jsonl"]:
                if os.path.exists(cand):
                    candidate_base = cand
                    break

        if candidate_base and os.path.exists(candidate_base):
            temp_samples = []
            try:
                with open(candidate_base, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            temp_samples.append(json.loads(line))
                        except Exception:
                            continue
            except Exception as e:
                print(f"[Warning] Failed to inspect local base '{candidate_base}': {e}")
                temp_samples = []

            if len(temp_samples) >= 2000:
                reusing_local_v1 = True
                print(
                    f"[✓] Local v1.0 base detected ({len(temp_samples)} samples)! "
                    f"Reusing local GooseReason and downloading only the 4 new v1.1 datasets..."
                )
                for item in temp_samples:
                    if scrub_refusals:
                        scrubbed = scrub_refusal_and_boilerplate(item)
                        if not scrubbed:
                            continue
                        item = scrubbed
                    if enforce_think:
                        msgs = item.get("messages", [])
                        for m in msgs:
                            if m.get("role") == "assistant":
                                m["content"] = format_to_cyberrott_cot(m.get("content", ""), enforce_think=True)
                    local_base_samples.append(item)

    # 3. Stream Datasets from Hugging Face Hub
    cyber_regex = compile_cyber_regex(CYBER_KEYWORDS)
    hf_samples = []

    if preset == "v1.1":
        active_keys = dataset_keys or list(CYBERROTT_V1_1_DATASETS.keys())
        if reusing_local_v1:
            active_keys = [k for k in active_keys if k != "goose_reason"]
            print(f"[*] Local reuse active: GooseReason omitted. Remaining datasets: {active_keys}")
        else:
            print(f"[*] Mode v1.1 Active: Processing {len(active_keys)} datasets from catalog...")

        if active_keys:
            hf_samples = fetch_multi_hf_datasets(
                dataset_keys=active_keys,
                cyber_regex=cyber_regex,
                scrub_refusals=scrub_refusals,
                enforce_think=enforce_think
            )
    else:
        target_repo = hf_dataset or "nvidia/Nemotron-Research-GooseReason-0.7M"
        limit = goose_limit if goose_limit is not None else max_hf_samples
        if limit > 0:
            print(f"[*] Classic v1.0 Mode: Ingestion of '{target_repo}' (max: {limit})...")
            try:
                hf_samples = fetch_single_dataset_stream(
                    dataset_name=target_repo,
                    max_samples=limit,
                    cyber_regex=cyber_regex,
                    scrub_refusals=scrub_refusals,
                    enforce_think=enforce_think
                )
            except Exception as e:
                print(f"[Warning] Connection failure or error from Hugging Face: {e}")

    # 4. Combine and Deduplicate
    combined = anchor_samples + local_base_samples + hf_samples
    if not combined:
        print("[Critical Error] No samples available to compose training dataset!")
        return []

    deduped_samples = deduplicate_samples(combined, preserve_anchors=True)

    # 5. Deterministic Shuffling
    random.seed(seed)
    random.shuffle(deduped_samples)

    # 6. Save to Disk
    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f_out:
        for entry in deduped_samples:
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 7. Final Consolidated Report
    print("\n" + "=" * 75)
    print("📊 CONSOLIDATED TRAINING DATASET REPORT")
    print("=" * 75)
    print(f"• Executed Preset:                          {preset.upper()}")
    print(f"• Unique Golden Anchors:                    {unique_anchor_count}")
    print(f"• Anchor Upsampling Factor:                 {anchor_repeat}x")
    print(f"• Total Anchor Samples in Training:         {len(anchor_samples)}")
    if reusing_local_v1:
        print(f"• Reused Local Base Samples:                {len(local_base_samples)}")
    print(f"• Total Hugging Face Samples Ingested:      {len(hf_samples)}")
    print(f"• FINAL TOTAL DEDUPLICATED AND SHUFFLED:    {len(deduped_samples)}")
    print(f"• Output File:                              {output_file}")
    print("=" * 75 + "\n")

    return deduped_samples


run_fusion_pipeline = prepare_dataset


# =============================================================================
# 11. COMMAND LINE INTERFACE (CLI)
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CyberRott Training Data Preparation and Fusion Pipeline (v1.0 & v1.1)"
    )

    parser.add_argument(
        "--preset",
        choices=["v1.0", "v1.1"],
        default="v1.0",
        help="Dataset preset: 'v1.0' (GooseReason classic 2.5k) or 'v1.1' (Multi-Dataset ~10k CoT)"
    )
    parser.add_argument(
        "--list_datasets",
        action="store_true",
        help="Displays the official v1.1 dataset catalog and exits"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Custom list of catalog keys to ingest (e.g. goose_reason code_feedback cyber_native)"
    )

    parser.add_argument(
        "--anchor_file", "--anchor-file",
        dest="anchor_file",
        default="dataset_cyber_ready.jsonl",
        help="Path to JSONL file containing the 16 complex Golden Anchors"
    )
    parser.add_argument(
        "--anchor_repeat", "--anchor-repeat",
        dest="anchor_repeat",
        type=int,
        default=20,
        help="Upsampling factor for Golden Anchors (default: 20 -> 320 samples)"
    )
    parser.add_argument(
        "--hf_dataset", "--hf-dataset",
        dest="hf_dataset",
        default="nvidia/Nemotron-Research-GooseReason-0.7M",
        help="Hugging Face dataset repository for streaming (v1.0 mode or override)"
    )
    parser.add_argument(
        "--max_hf_samples", "--max-hf-samples",
        dest="max_hf_samples",
        type=int,
        default=2500,
        help="Maximum samples to stream from Hugging Face in v1.0 mode"
    )
    parser.add_argument(
        "--output", "--output_file", "--output-file",
        dest="output",
        default="cyberrott_final_train.jsonl",
        help="Target output JSONL file path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling"
    )

    parser.add_argument(
        "--scrub_refusals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable refusal and disclaimer scrubber"
    )
    parser.add_argument(
        "--enforce_think_tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce and structure <think>...</think> CoT blocks across all assistant responses"
    )

    parser.add_argument(
        "--goose_limit", "--goose-limit",
        dest="goose_limit",
        type=int,
        default=None,
        help="Custom sample limit for GooseReason (overrides default target_samples for goose_reason)"
    )

    args = parser.parse_args()

    if args.goose_limit is not None:
        CYBERROTT_V1_1_DATASETS["goose_reason"]["target_samples"] = args.goose_limit

    if args.list_datasets:
        print_dataset_catalog()
        sys.exit(0)

    prepare_dataset(
        anchor_file=args.anchor_file,
        anchor_repeat=args.anchor_repeat,
        preset=args.preset,
        dataset_keys=args.datasets,
        hf_dataset=args.hf_dataset,
        max_hf_samples=args.max_hf_samples,
        scrub_refusals=args.scrub_refusals,
        enforce_think=args.enforce_think_tags,
        output_file=args.output,
        seed=args.seed,
        goose_limit=args.goose_limit
    )
