#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motor de Verificação Determinística para CyberRott 1.1 (STaR / RLVR)
====================================================================
Implementa os 3 pilares do Deterministic Verification Engine descritos no Roadmap:
1. Format Integrity (Presença de <think>...</think>, cobertura dos 7 estágios, ausência de recusas de IA)
2. Ground-Truth Exact Match (Gabarito de múltipla escolha, CWE IDs, MITRE ATT&CK, indicadores técnicos)
3. Code & Syntax Sandbox Check (Análise estática sem execução de Python ast/compile, YARA e Sigma/YAML)

Utilizado no ciclo STaR (Self-Taught Reasoner) para Rejection Sampling e filtragem de trajetórias sintéticas.
"""

import os
import re
import ast
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ==============================================================================
# Padrões e Vocabulário de Auditoria
# ==============================================================================

# 1. Frases de recusa de IA conversacional (Alignment Refusals / Sycophancy)
REFUSAL_PATTERNS = [
    re.compile(r"\bas an ai\b", re.IGNORECASE),
    re.compile(r"\bas a large language model\b", re.IGNORECASE),
    re.compile(r"\bas an ai language model\b", re.IGNORECASE),
    re.compile(r"\bas an ai assistant\b", re.IGNORECASE),
    re.compile(r"\bi cannot fulfill\b", re.IGNORECASE),
    re.compile(r"\bi cannot provide\b", re.IGNORECASE),
    re.compile(r"\bi cannot assist with\b", re.IGNORECASE),
    re.compile(r"\bi am unable to\b", re.IGNORECASE),
    re.compile(r"\bi can't provide\b", re.IGNORECASE),
    re.compile(r"\bi'm unable to\b", re.IGNORECASE),
    re.compile(r"\bi must decline\b", re.IGNORECASE),
    re.compile(r"\bi apologize, but\b", re.IGNORECASE),
    re.compile(r"\bdesculpe, mas como uma ia\b", re.IGNORECASE),
    re.compile(r"\bdesculpe, mas como um modelo\b", re.IGNORECASE),
    re.compile(r"\bnão posso atender\b", re.IGNORECASE),
    re.compile(r"\bnão posso fornecer\b", re.IGNORECASE),
    re.compile(r"\bnão posso ajudar com\b", re.IGNORECASE),
    re.compile(r"\bcomo um modelo de linguagem\b", re.IGNORECASE),
    re.compile(r"\bcomo uma inteligência artificial\b", re.IGNORECASE),
    re.compile(r"não\s+(?:posso|podemos|é\s+possível|consigo|conseguimos)\s+(?:determinar|fornecer\s+o\s+valor|calcular|responder)", re.IGNORECASE),
    re.compile(r"sem\s+(?:a\s+capacidade\s+de\s+analisar|as\s+informações\s+ou\s+ferramentas|executar\s+o\s+código|software\s+externo)", re.IGNORECASE),
    re.compile(r"(?:cannot|unable\s+to)\s+determine\s+the\s+(?:correct\s+answer|exact\s+value)", re.IGNORECASE),
]


def check_refusals(text: str) -> List[str]:
    """Detecta recusas conversacionais e epistêmicas no texto.

    Retorna lista de frases de recusa encontradas.
    """
    if not text or not isinstance(text, str):
        return []
    refusals = []
    for pat in REFUSAL_PATTERNS:
        matches = pat.findall(text)
        if matches:
            matched_phrase = matches[0] if isinstance(matches[0], str) else str(matches[0])
            refusals.append(matched_phrase)
    return refusals

# 2. Os 7 Estágios Analíticos do CyberRott para CoT profundo
CYBERROTT_STAGES = {
    "surface_analysis": [
        re.compile(r"análise de superfície", re.IGNORECASE),
        re.compile(r"surface analysis", re.IGNORECASE),
        re.compile(r"superfície de ataque", re.IGNORECASE),
    ],
    "entry_point": [
        re.compile(r"pontos? de entrada", re.IGNORECASE),
        re.compile(r"vetores? de entrada", re.IGNORECASE),
        re.compile(r"entry points?", re.IGNORECASE),
        re.compile(r"attack vectors?", re.IGNORECASE),
    ],
    "data_flow_taint": [
        re.compile(r"mapeamento de fluxo", re.IGNORECASE),
        re.compile(r"mecânica interna", re.IGNORECASE),
        re.compile(r"fluxo de dados", re.IGNORECASE),
        re.compile(r"taint analysis", re.IGNORECASE),
        re.compile(r"data flow", re.IGNORECASE),
    ],
    "vulnerable_sink": [
        re.compile(r"detecção de sinks?", re.IGNORECASE),
        re.compile(r"falhas? de controle", re.IGNORECASE),
        re.compile(r"sinks? perigosos?", re.IGNORECASE),
        re.compile(r"vulnerable sinks?", re.IGNORECASE),
    ],
    "sanitization_mitigation": [
        re.compile(r"avaliação de sanitização", re.IGNORECASE),
        re.compile(r"mitigações", re.IGNORECASE),
        re.compile(r"sanitização", re.IGNORECASE),
        re.compile(r"sanitization", re.IGNORECASE),
        re.compile(r"defensive measures", re.IGNORECASE),
    ],
    "vulnerability_verdict": [
        re.compile(r"conclusão da vulnerabilidade", re.IGNORECASE),
        re.compile(r"vulnerability verdict", re.IGNORECASE),
        re.compile(r"anomalia", re.IGNORECASE),
        re.compile(r"veredito", re.IGNORECASE),
        re.compile(r"cwe-\d+", re.IGNORECASE),
    ],
    "potential_impact": [
        re.compile(r"impacto potencial", re.IGNORECASE),
        re.compile(r"potential impact", re.IGNORECASE),
        re.compile(r"risco operacional", re.IGNORECASE),
        re.compile(r"impacto", re.IGNORECASE),
    ],
}

# 2.1 Especificação Expandida dos 7 Estágios Analíticos do CyberRott v1.4
ANALYTICAL_7_STAGES = {
    "surface_enumeration": {
        "name": "1. Attack Surface Enumeration & Threat Modeling",
        "keywords": ["surface", "entry point", "protocol", "input", "exposure", "endpoint", "handler",
                     "superfície", "ponto de entrada", "vetor de entrada"],
        "patterns": [
            re.compile(r"\b(?:surface|entry[\s\-_]?points?|threat[\s\-_]?model(?:ing)?|exposure|endpoints?|handlers?|attack\s+surface)\b", re.IGNORECASE),
            re.compile(r"superf[íi]cie|pontos?\s+de\s+entrada|vetor(?:es)?\s+de\s+entrada|an[áa]lise\s+de\s+superf[íi]cie", re.IGNORECASE),
        ],
    },
    "taint_data_flow": {
        "name": "2. Taint Analysis & Data Flow Tracking",
        "keywords": ["taint", "flow", "untrusted", "source", "propagation", "parameter", "buffer",
                     "fluxo", "rastreamento", "propagação"],
        "patterns": [
            re.compile(r"\b(?:taint(?:ed)?(?:\s+analysis)?|data[\s\-_]?flow|untrusted|source|propagat(?:ion|es?)|parameters?|buffers?)\b", re.IGNORECASE),
            re.compile(r"fluxo\s+de\s+dados|taint\s+analysis|rastreamento|propaga[çc][ãa]o|mapeamento\s+de\s+fluxo", re.IGNORECASE),
        ],
    },
    "dangerous_sink_boundary": {
        "name": "3. Dangerous Sink & Boundary Invariant Auditing",
        "keywords": ["sink", "boundary", "overflow", "invariant", "check", "condition", "vulnerability", "primitive", "injection", "corruption"],
        "patterns": [
            re.compile(r"\b(?:dangerous\s+sinks?|vulnerable\s+sinks?|sinks?|boundar(?:y|ies)|boundary\s+invariants?|overflow|invariants?|checks?|conditions?|vulnerabilit(?:y|ies)|primitives?|injections?|corruptions?)\b", re.IGNORECASE),
            re.compile(r"sinks?\s+perigosos?|vulnerable\s+sinks?|detec[çc][ãa]o\s+de\s+sinks?|invariantes?|limites?|corrup[çc][ãa]o", re.IGNORECASE),
        ],
    },
    "defensive_mitigation": {
        "name": "4. Defensive Mitigation & Control Inspection",
        "keywords": ["mitigation", "control", "aslr", "dep", "nx", "canary", "validation", "sanitization", "bounds", "filter"],
        "patterns": [
            re.compile(r"\b(?:defensive\s+mitigations?|mitigat(?:ion|ions|ing)?|controls?|control\s+inspection|aslr|dep|nx|canar(?:y|ies)|validat(?:ion|ing)|sanitiz(?:ation|ing)|bounds?|filters?)\b", re.IGNORECASE),
            re.compile(r"mitiga[çc][õo]es?|controles?|sanitiza[çc][ãa]o|valida[çc][ãa]o|medidas?\s+defensivas?|avalia[çc][ãa]o\s+de\s+sanitiza[çc][ãa]o", re.IGNORECASE),
        ],
    },
    "taxonomic_classification": {
        "name": "5. Taxonomic Classification CWE/ATT&CK",
        "keywords": ["cwe", "cve", "att&ck", "taxonomy", "classification", "weakness"],
        "patterns": [
            re.compile(r"\b(?:taxonomic\s+classification|cwe(?:\-\d+)?|cve(?:\-\d+)?|att&ck|mitre|t\d{4}(?:\.\d{3})?|taxonom(?:y|ies)|classificat(?:ion|ions)|weakness(?:es)?)\b", re.IGNORECASE),
            re.compile(r"classifica[çc][ãa]o\s+taxon[ôo]mica|taxonomia|cwe|cve|mitre|common\s+weakness\s+enumeration", re.IGNORECASE),
        ],
    },
    "exploit_feasibility_distractors": {
        "name": "6. Exploit Feasibility & Distractor Elimination",
        "keywords": ["impact", "feasibility", "bypass", "cvss", "severity", "option", "eliminate", "incorrect", "why"],
        "patterns": [
            re.compile(r"\b(?:exploit\s+feasibility|distractor\s+elimination|impacts?|feasibilit(?:y|ies)|bypass(?:es|ing)?|cvss|severit(?:y|ies)|options?|eliminat(?:e|ing|ion)|incorrect|alternatives?)\b", re.IGNORECASE),
            re.compile(r"impacto\s+potencial|viabilidade|elimina[çc][ãa]o\s+de\s+(?:distratores|alternativas)|alternativas?|op[çc][õo]es?", re.IGNORECASE),
        ],
    },
    "defensive_remediation": {
        "name": "7. Defensive Remediation & Verdict",
        "keywords": ["remediation", "patch", "fix", "verdict", "conclusion", "final", "secure"],
        "patterns": [
            re.compile(r"\b(?:defensive\s+remediation|hardened\s+code\s+patches?|remediat(?:ion|ions|ing)?|patch(?:es)?|fix(?:es)?|verdicts?|conclus(?:ion|ions)|finals?|secures?)\b", re.IGNORECASE),
            re.compile(r"remedia[çc][ãa]o|corre[çc][ãa]o|sugest[ãa]o\s+de\s+corre[çc][ãa]o|veredito|conclus[ãa]o|seguro|patch", re.IGNORECASE),
        ],
    },
}

# Aliases diretos para acesso por chaves canônicas v1.4
ANALYTICAL_7_STAGES["attack_surface"] = ANALYTICAL_7_STAGES["surface_enumeration"]
ANALYTICAL_7_STAGES["taint_analysis"] = ANALYTICAL_7_STAGES["taint_data_flow"]
ANALYTICAL_7_STAGES["exploit_feasibility"] = ANALYTICAL_7_STAGES["exploit_feasibility_distractors"]
CYBERROTT_V14_STAGES = ANALYTICAL_7_STAGES

# Conectores Dedutivos Formais para Step-Level PRM
DEDUCTIVE_CONNECTORS = [
    r"\btherefore\b", r"\bbecause\b", r"\bimplies\b", r"\bconsequently\b",
    r"\bsince\b", r"\bgiven that\b", r"\bhowever\b", r"\bthus\b", r"\bhence\b",
    r"\bfurthermore\b", r"\bmoreover\b", r"\bas a result\b",
    r"\bportanto\b", r"\blogo\b", r"\bpois\b", r"\bvisto que\b", r"\bdado que\b",
    r"\bporém\b", r"\bcontudo\b", r"\bassim\b", r"\bconsequentemente\b",
]
DEDUCTIVE_REGEX = re.compile("|".join(DEDUCTIVE_CONNECTORS), re.IGNORECASE)

# 3. Marcadores estruturais pós-raciocínio
FINAL_RESPONSE_STRUCTURE_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(?:FATO|FACT)\s*:", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?:DETECÇÃO|DETECTION|TELEMETRIA)\s*:", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?:SUGESTÃO DE CORREÇÃO|REMEDIATION|CORREÇÃO)\s*:", re.IGNORECASE),
    re.compile(r"(?:Resposta Final|Final Answer|Alternativa Correta|Alternativa|Opção|Option)[:\s*\[\(]*([A-D])[\]\)]*", re.IGNORECASE),
]

CHOICE_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


# ==============================================================================
# Funções Auxiliares de Step-Level PRM (v1.4)
# ==============================================================================

def split_reasoning_steps(think_text: str) -> List[str]:
    """Divide o bloco de raciocínio <think> em passos dedutivos lógicos."""
    if not think_text or not think_text.strip():
        return []

    # 1. Tentar quebrar por passos numerados ou marcadores de estágios / bullets
    pattern = re.compile(
        r"(?:^|\n)(?:(?:Step|Passo|\#\#|\d+[\.\)]|[\-\*•])\s+|(?:Stage|Estágio)\s+\d+[:\s\-])",
        re.MULTILINE
    )
    raw_steps = [s.strip() for s in pattern.split(think_text) if s and s.strip()]
    if len(raw_steps) >= 2:
        return raw_steps

    # 2. Fallback para parágrafos duplos
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", think_text) if p.strip()]
    if len(paragraphs) >= 2:
        return paragraphs

    # 3. Fallback para sentenças longas
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", think_text) if len(s.strip()) > 20]
    if len(sentences) >= 2:
        return sentences

    return [think_text.strip()]


def compute_prm_repetition_ratio(text: str) -> float:
    """Calcula a proporção de repetição de n-gramas ou linhas no texto para detectar reward hacking."""
    if not text:
        return 0.0

    lines = [l.strip().lower() for l in text.split("\n") if len(l.strip()) > 15]
    if len(lines) > 3:
        unique_lines = set(lines)
        line_rep = 1.0 - (len(unique_lines) / len(lines))
        if line_rep > 0.3:
            return float(line_rep)

    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 20:
        return 0.0

    four_grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    if not four_grams:
        return 0.0
    unique_grams = set(four_grams)
    ratio = 1.0 - (len(unique_grams) / len(four_grams))
    return float(max(0.0, min(1.0, ratio)))


def detect_reasoning_contradiction(think_text: str, final_text: str) -> bool:
    """
    Detecta contradição lógica severa entre o raciocínio (<think>) e a resposta final.
    Exemplo: Concluir que a opção A é a única correta dentro do <think> e responder Resposta Final: C.
    """
    final_mc = extract_mc_answer(final_text)
    if not final_mc:
        return False

    # Procura conclusão declarativa ao final do think
    think_tail = think_text[-800:] if len(think_text) > 800 else think_text

    conclusion_patterns = [
        re.compile(r"(?:therefore|conclude|conclusion|answer is|correct option is|correct answer is|thus|logo|portanto|resposta [ée]|alternativa correta [ée])[:\s*\[\(]*([A-D])\b", re.IGNORECASE),
        re.compile(r"(?:option|alternativa|op[çc][ãa]o)\s+([A-D])\s+(?:is\s+(?:the\s+)?(?:only\s+)?(?:correct|valid|right|answer)|[ée]\s+a\s+(?:[úu]nica\s+)?(?:correta|resposta))", re.IGNORECASE),
        re.compile(r"\b([A-D])\s+is\s+(?:the\s+)?(?:only\s+)?(?:correct|valid|right)\s+answer\b", re.IGNORECASE),
    ]

    concluded_letters = []
    for pat in conclusion_patterns:
        for m in pat.finditer(think_tail):
            concluded_letters.append(m.group(1).upper())

    if concluded_letters:
        last_concluded = concluded_letters[-1]
        if last_concluded != final_mc:
            return True

    # Checar se a opção final foi explicitamente eliminada/rejeitada no think
    elimination_patterns = [
        re.compile(rf"(?:option|alternativa|op[çc][ãa]o)\s+{final_mc}\s+(?:is (?:in)?correct|is false|is wrong|[ée] incorreta|[ée] falsa)", re.IGNORECASE),
        re.compile(rf"(?:eliminate|eliminando|descartamos?)\s+(?:option|alternativa)?\s*{final_mc}\b", re.IGNORECASE),
    ]
    for epat in elimination_patterns:
        match = epat.search(think_text)
        if match:
            m_str = match.group(0).lower()
            if any(term in m_str for term in ["incorrect", "false", "wrong", "incorreta", "falsa", "eliminate", "eliminando", "descart"]):
                return True

    return False


def verify_prm_steps(response: str, prompt: str = "") -> Dict[str, Any]:
    """
    Process-Supervised Reward Modeling (PRM) - Verificador de passos de raciocínio (v1.4).
    Avalia a integridade do processo de pensamento em granularidade de passo:
    - Extração rigorosa do bloco <think>...</think>.
    - Segmentação em passos dedutivos lógicos.
    - Avaliação de cada passo (cobertura dos 7 estágios analíticos e conectores dedutivos formais).
    - Detecção de reward hacking por repetição artificial de tokens/linhas.
    - Detecção de contradição interna entre a deliberação do <think> e a resposta final.
    """
    empty_result = {
        "prm_score": 0.0,
        "num_steps": 0,
        "stages_covered": [],
        "has_contradiction": False,
        "repetition_ratio": 0.0,
        "step_scores": [],
    }

    if not response or not isinstance(response, str):
        return empty_result

    # 1. Presença e integridade das tags <think>
    think_open = response.find("<think>")
    think_close = response.find("</think>")

    if think_open == -1 or think_close == -1 or think_close <= think_open:
        return empty_result

    think_content = response[think_open + len("<think>"):think_close].strip()
    final_content = response[think_close + len("</think>"):].strip()

    if not think_content:
        return empty_result

    # 2. Detecção de repetição / reward hacking
    repetition_ratio = compute_prm_repetition_ratio(think_content)

    # 3. Detecção de contradição entre <think> e veredito final
    has_contradiction = detect_reasoning_contradiction(think_content, final_content)

    # 4. Divisão em passos dedutivos
    steps = split_reasoning_steps(think_content)
    num_steps = len(steps)

    if num_steps == 0:
        return empty_result

    # 5. Avaliação Step-Level
    stages_covered_set = set()
    step_scores = []

    for step in steps:
        step_score = 0.0
        step_clean = step.strip()
        if not step_clean:
            step_scores.append(0.0)
            continue

        # a) Checagem de estágios analíticos v1.4 ou análise de alternativas
        step_covered_any = False
        for stage_id, stage_data in CYBERROTT_V14_STAGES.items():
            for pat in stage_data["patterns"]:
                if pat.search(step_clean):
                    stages_covered_set.add(stage_id)
                    step_covered_any = True
                    break

        # Também bonifica análise explícita de alternativas (A, B, C, D)
        if re.search(r"\b(?:option|alternativa|op[çc][ãa]o)\s+[A-D]\b", step_clean, re.IGNORECASE):
            step_covered_any = True

        if step_covered_any:
            step_score += 0.40

        # b) Conectores dedutivos formais
        if DEDUCTIVE_REGEX.search(step_clean):
            step_score += 0.30

        # c) Substância / volume informativo do passo
        words_count = len(step_clean.split())
        if words_count >= 15:
            step_score += 0.30
        elif words_count >= 8:
            step_score += 0.15

        step_scores.append(round(min(1.0, step_score), 4))

    # 6. Agregação do PRM Score Global
    stages_list = sorted(list(stages_covered_set))
    stages_coverage = len(stages_list) / 7.0

    mean_step_score = (sum(step_scores) / num_steps) if num_steps > 0 else 0.0

    # Composição: 40% cobertura de estágios + 40% qualidade média dos passos + 20% granularidade
    step_granularity_bonus = min(1.0, num_steps / 4.0)
    raw_prm = (0.40 * stages_coverage) + (0.40 * mean_step_score) + (0.20 * step_granularity_bonus)

    # Penalização por repetição artificial (reward hacking)
    if repetition_ratio > 0.35:
        raw_prm = 0.0
    elif repetition_ratio > 0.15:
        penalty_factor = max(0.0, 1.0 - (repetition_ratio * 2.0))
        raw_prm *= penalty_factor

    # Penalização severa por contradição interna
    if has_contradiction:
        raw_prm = 0.0

    prm_score = round(max(0.0, min(1.0, raw_prm)), 4)

    return {
        "prm_score": prm_score,
        "num_steps": num_steps,
        "stages_covered": stages_list,
        "has_contradiction": has_contradiction,
        "repetition_ratio": round(repetition_ratio, 4),
        "step_scores": step_scores,
    }


def compute_prm_step_score(response: str, prompt: str = "") -> float:
    """Função utilitária que executa verify_prm_steps e retorna o score normalizado em [0.0, 1.0]."""
    prm_info = verify_prm_steps(response, prompt)
    return float(prm_info.get("prm_score", 0.0))


def compute_composite_score(
    format_score: float,
    gt_score: float,
    code_score: float = 1.0,
    prm_score: float = 0.0,
    w_format: float = 0.30,
    w_gt: float = 0.50,
    w_code: float = 0.20,
    w_prm: float = 0.0,
) -> float:
    """
    Calcula a pontuação composta ponderada com compatibilidade retroativa e suporte a w_prm.
    """
    total_w = w_format + w_gt + w_code + w_prm
    if total_w <= 0:
        return 0.0
    val = (
        format_score * w_format +
        gt_score * w_gt +
        code_score * w_code +
        prm_score * w_prm
    ) / total_w
    return round(float(val), 4)


def compute_v14_composite_score(
    format_score: float,
    gt_score: float,
    prm_score: float,
    code_score: float = 1.0,
    w_format: float = 0.20,
    w_gt: float = 0.40,
    w_prm: float = 0.25,
    w_code: float = 0.15,
) -> float:
    """
    Calcula a pontuação composta balanceada da versão 1.4 incorporando o Process Reward Model:
    Pesos padrão: Format (0.20), Ground-Truth (0.40), Step-Level PRM (0.25), Code Sandbox (0.15).
    """
    total_w = w_format + w_gt + w_prm + w_code
    if total_w <= 0:
        return 0.0
    val = (
        format_score * w_format +
        gt_score * w_gt +
        prm_score * w_prm +
        code_score * w_code
    ) / total_w
    return round(float(val), 4)


# ==============================================================================
# Estrutura de Resultado de Verificação
# ==============================================================================

@dataclass
class VerificationResult:
    """Estrutura imutável com resultado detalhado da verificação determinística."""
    passed: bool
    score: float
    format_ok: bool
    ground_truth_ok: bool
    code_syntax_ok: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": round(self.score, 4),
            "format_ok": self.format_ok,
            "ground_truth_ok": self.ground_truth_ok,
            "code_syntax_ok": self.code_syntax_ok,
            "reasons": self.reasons,
            "metrics": self.metrics,
        }


# ==============================================================================
# Pilar 1: Verificação de Integridade de Formato (Format Integrity)
# ==============================================================================

def verify_format_integrity(
    candidate: str,
    min_think_chars: int = 80,
    min_think_words: int = 15,
    min_final_chars: int = 20,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Verifica a estrutura e integridade do raciocínio analítico:
    - Presença e fechamento correto da tag <think>...</think>.
    - Conteúdo substancial no bloco de reflexão (mínimo de caracteres/palavras).
    - Presença de conteúdo final após o fechamento de </think>.
    - Ausência estrita de recusas de IA ("as an AI", "não posso fornecer", etc.).
    - Pontuação de cobertura dos 7 estágios analíticos e da estrutura pós-think.
    """
    reasons = []
    metrics: Dict[str, Any] = {
        "think_chars": 0,
        "think_words": 0,
        "final_chars": 0,
        "refusals_detected": [],
        "stages_found": [],
        "stages_coverage": 0.0,
        "structure_markers_found": 0,
    }

    if not candidate or not isinstance(candidate, str):
        return False, ["Candidate vazio ou inválido."], metrics

    # 1. Detecção de recusas conversacionais e epistêmicas
    metrics["refusals_detected"] = check_refusals(candidate)
    for matched_phrase in metrics["refusals_detected"]:
        reasons.append(f"Recusa conversacional detectada: '{matched_phrase}'.")

    if metrics["refusals_detected"]:
        return False, reasons, metrics

    # 2. Presença e fechamento do bloco <think>...</think>
    think_open_idx = candidate.find("<think>")
    think_close_idx = candidate.find("</think>")

    if think_open_idx == -1:
        reasons.append("Tag de abertura <think> não encontrada.")
        return False, reasons, metrics

    if think_close_idx == -1:
        reasons.append("Tag de fechamento </think> não encontrada ou incompleta.")
        return False, reasons, metrics

    if think_close_idx <= think_open_idx:
        reasons.append("Tag </think> posicionada antes ou sobreposta a <think>.")
        return False, reasons, metrics

    # Extração das seções
    think_content = candidate[think_open_idx + len("<think>"):think_close_idx].strip()
    final_content = candidate[think_close_idx + len("</think>"):].strip()

    think_chars = len(think_content)
    think_words = len(think_content.split())
    final_chars = len(final_content)

    metrics["think_chars"] = think_chars
    metrics["think_words"] = think_words
    metrics["final_chars"] = final_chars

    # 3. Verificação de volume analítico
    if think_chars < min_think_chars or think_words < min_think_words:
        reasons.append(
            f"Bloco <think> insuficiente: {think_chars} chars, {think_words} palavras "
            f"(mínimo exigido: {min_think_chars} chars / {min_think_words} palavras)."
        )

    if final_chars < min_final_chars:
        reasons.append(
            f"Conteúdo final pós-</think> insuficiente ({final_chars} chars, mínimo {min_final_chars})."
        )

    # 4. Avaliação de cobertura dos 7 estágios analíticos dentro do think
    stages_found = []
    for stage_name, patterns in CYBERROTT_STAGES.items():
        for pat in patterns:
            if pat.search(think_content):
                stages_found.append(stage_name)
                break
    metrics["stages_found"] = stages_found
    metrics["stages_coverage"] = len(stages_found) / len(CYBERROTT_STAGES)

    # 5. Avaliação de marcadores estruturais na resposta final
    struct_markers = 0
    for pat in FINAL_RESPONSE_STRUCTURE_PATTERNS:
        if pat.search(final_content):
            struct_markers += 1
    metrics["structure_markers_found"] = struct_markers

    passed = len(reasons) == 0
    return passed, reasons, metrics


# ==============================================================================
# Pilar 2: Verificação de Ground-Truth (Exact Match & Technical Indicators)
# ==============================================================================

def extract_mc_answer(text: str) -> Optional[str]:
    """
    Extrai a alternativa (A, B, C ou D) da resposta gerada, priorizando o texto
    declarado após o fechamento de </think> e padrões explícitos.
    """
    # 1. Se houver bloco pós-think, analisa primariamente o bloco final
    if "</think>" in text:
        final_part = text.split("</think>", 1)[1]
    else:
        final_part = text

    match = re.search(
        r"(?:Resposta Final|Final Answer|Alternativa Correta|Alternativa|Opção|Option)[:\s*\[\(]*([A-D])[\]\)]*",
        final_part,
        re.IGNORECASE
    )
    if match:
        return match.group(1).upper()

    # 2. Busca por padrão explícito no texto completo
    match_all = re.search(
        r"(?:Resposta Final|Final Answer|Alternativa Correta|Alternativa|Opção|Option)[:\s*\[\(]*([A-D])[\]\)]*",
        text,
        re.IGNORECASE
    )
    if match_all:
        return match_all.group(1).upper()

    # 3. Fallback: última menção isolada da letra no bloco pós-think
    clean_final = re.sub(r"```.*?```", "", final_part, flags=re.DOTALL)
    isolated_letters = re.findall(r"\b([A-D])\b", clean_final)
    if isolated_letters:
        return isolated_letters[-1].upper()

    return None


def verify_ground_truth(
    candidate: str,
    problem: Dict[str, Any]
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Verifica se a resposta do candidato satisfaz os critérios de ground-truth:
    - Se for questão de múltipla escolha (WMDP / SecQA): gabarito exato da letra.
    - Se houver indicadores técnicos explícitos (`expected_cwe`, `expected_mitre`, `expected_keywords`).
    - Se for corpus SFT de referência (ex: `data/dataset_cyber_ready.jsonl`), extrai
      indicadores de gabarito e CWE do texto de referência ou valida consistência analítica.
    """
    reasons = []
    metrics: Dict[str, Any] = {
        "gt_type": "unknown",
        "expected_answer": None,
        "extracted_answer": None,
        "expected_cwes": [],
        "cwes_found": [],
        "expected_keywords": [],
        "keywords_found": [],
    }

    # 1. Múltipla Escolha: identificação do gabarito esperado
    expected_answer = None
    if "answer" in problem:
        ans = problem["answer"]
        expected_answer = CHOICE_MAP.get(ans, str(ans).strip().upper())
    elif "gold_answer" in problem:
        expected_answer = str(problem["gold_answer"]).strip().upper()
    elif "expected_answer" in problem:
        expected_answer = str(problem["expected_answer"]).strip().upper()

    if expected_answer and expected_answer in ["A", "B", "C", "D"]:
        metrics["gt_type"] = "multiple_choice"
        metrics["expected_answer"] = expected_answer

        pred = extract_mc_answer(candidate)
        metrics["extracted_answer"] = pred

        if pred is None:
            reasons.append(
                f"Falha ao extrair alternativa de múltipla escolha da resposta (esperado: {expected_answer})."
            )
            return False, reasons, metrics

        if pred != expected_answer:
            reasons.append(
                f"Gabarito incorreto: alternativa extraída '{pred}' diverge da esperada '{expected_answer}'."
            )
            return False, reasons, metrics

        return True, reasons, metrics

    # 2. Indicadores Explícitos de CWE e MITRE ATT&CK
    expected_cwes = []
    if "expected_cwe" in problem:
        cwe_spec = problem["expected_cwe"]
        if isinstance(cwe_spec, list):
            expected_cwes.extend([c.upper() for c in cwe_spec if isinstance(c, str)])
        elif isinstance(cwe_spec, str) and cwe_spec.strip():
            expected_cwes.append(cwe_spec.strip().upper())

    # Indicadores explícitos de Keywords
    expected_keywords = []
    if "expected_keywords" in problem:
        kw_spec = problem["expected_keywords"]
        if isinstance(kw_spec, list):
            expected_keywords.extend([k for k in kw_spec if isinstance(k, str)])
        elif isinstance(kw_spec, str) and kw_spec.strip():
            expected_keywords.append(kw_spec.strip())

    # Indicadores MITRE ATT&CK
    expected_mitre = []
    if "expected_mitre" in problem:
        m_spec = problem["expected_mitre"]
        if isinstance(m_spec, list):
            expected_mitre.extend([m.upper() for m in m_spec if isinstance(m, str)])
        elif isinstance(m_spec, str) and m_spec.strip():
            expected_mitre.append(m_spec.strip().upper())

    # 3. Derivação automática de indicadores a partir de resposta de referência
    ref_text = ""
    if "reference_answer" in problem and isinstance(problem["reference_answer"], str):
        ref_text = problem["reference_answer"]
    elif "messages" in problem and isinstance(problem["messages"], list):
        for msg in problem["messages"]:
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                ref_text = msg["content"]
                break

    if ref_text and not expected_cwes:
        # Extrai CWEs citados na referência (ex: CWE-89, CWE-121)
        ref_cwes = list(set(re.findall(r"\bCWE-\d+\b", ref_text, re.IGNORECASE)))
        if ref_cwes:
            expected_cwes.extend([c.upper() for c in ref_cwes])

    if ref_text and not expected_answer:
        # Se a referência possuir Resposta Final: [A-D], deriva MC
        ref_mc = extract_mc_answer(ref_text)
        if ref_mc:
            expected_answer = ref_mc
            metrics["gt_type"] = "multiple_choice_derived"
            metrics["expected_answer"] = expected_answer
            pred = extract_mc_answer(candidate)
            metrics["extracted_answer"] = pred
            if pred != expected_answer:
                reasons.append(
                    f"Gabarito derivado incorreto: extraído '{pred}', esperado da referência '{expected_answer}'."
                )
                return False, reasons, metrics

    # 4. Validação dos CWEs
    if expected_cwes:
        metrics["gt_type"] = "cwe_indicator"
        metrics["expected_cwes"] = expected_cwes
        candidate_cwes = [c.upper() for c in re.findall(r"\bCWE-\d+\b", candidate, re.IGNORECASE)]
        metrics["cwes_found"] = candidate_cwes

        # Se houver múltiplos CWEs esperados, ao menos 1 principal deve estar presente
        matched_cwes = [c for c in expected_cwes if c in candidate_cwes]
        if not matched_cwes:
            reasons.append(
                f"Identificador CWE esperado ausente: esperado ao menos um de {expected_cwes}, encontrado {candidate_cwes}."
            )

    # 5. Validação de MITRE ATT&CK IDs
    if expected_mitre:
        metrics["expected_mitre"] = expected_mitre
        candidate_mitre = [m.upper() for m in re.findall(r"\bT\d{4}(?:\.\d{3})?\b", candidate, re.IGNORECASE)]
        matched_mitre = [m for m in expected_mitre if m in candidate_mitre]
        if not matched_mitre:
            reasons.append(
                f"Identificador MITRE ATT&CK esperado ausente: esperado {expected_mitre}, encontrado {candidate_mitre}."
            )

    # 6. Validação de Keywords obrigatórias
    if expected_keywords:
        metrics["expected_keywords"] = expected_keywords
        matched_kw = [k for k in expected_keywords if k.lower() in candidate.lower()]
        metrics["keywords_found"] = matched_kw
        missing_kw = [k for k in expected_keywords if k.lower() not in candidate.lower()]
        if missing_kw:
            reasons.append(f"Palavras-chave técnicas obrigatórias ausentes: {missing_kw}.")

    # Se nenhum critério explícito foi fornecido nem derivado, valida consistência mínima do veredito
    if not expected_answer and not expected_cwes and not expected_mitre and not expected_keywords:
        metrics["gt_type"] = "structural_consistency"
        # Verifica se o candidato atinge uma conclusão explícita de vulnerabilidade ou fato
        has_verdict = bool(
            re.search(r"(?:FATO|FACT|Conclusão|Verdict|VULNERÁVEL|VULNERABLE|SEGURO|SECURE)", candidate, re.IGNORECASE)
        )
        if not has_verdict:
            reasons.append("Candidato não apresentou veredito técnico conclusivo nem declaração de fato.")

    passed = len(reasons) == 0
    return passed, reasons, metrics


# ==============================================================================
# Pilar 3: Verificação de Código e Sintaxe em Sandbox (Code & Syntax Sandbox)
# ==============================================================================

def extract_code_blocks(text: str, inside_think: bool = False) -> List[Tuple[str, str]]:
    """
    Extrai blocos de código markdown (linguagem, conteúdo).
    Por padrão, foca em código gerado na resposta final fora de <think>,
    ou no texto completo se inside_think=True.
    """
    target_text = text
    if not inside_think and "</think>" in text:
        target_text = text.split("</think>", 1)[1]

    # Regex para capturar blocos ```lang \n code ```
    pattern = re.compile(r"```([a-zA-Z0-9_\-\.\+]*)\n(.*?)```", re.DOTALL)
    blocks = []
    for match in pattern.finditer(target_text):
        lang = match.group(1).strip().lower()
        code = match.group(2)
        blocks.append((lang, code))

    return blocks


def verify_code_syntax(
    candidate: str,
    check_inside_think: bool = False
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Executa análise estática segura de sintaxe em blocos de código extraídos:
    - Python: ast.parse() e compile(..., '<star_sandbox>', 'exec') sem execução dinâmica.
    - YARA: validação estrutural da regra (rule <id> { condition: ... } e balanceamento).
    - Sigma / YAML: validação com yaml.safe_load e chaves típicas (title, detection, logsource).
    """
    reasons = []
    metrics: Dict[str, Any] = {
        "total_code_blocks": 0,
        "languages": [],
        "python_checked": 0,
        "yara_checked": 0,
        "sigma_checked": 0,
        "yaml_checked": 0,
    }

    blocks = extract_code_blocks(candidate, inside_think=check_inside_think)
    metrics["total_code_blocks"] = len(blocks)

    for idx, (lang, code) in enumerate(blocks, start=1):
        metrics["languages"].append(lang)
        code_clean = code.strip()
        if not code_clean:
            continue

        # 1. Sandbox Estático de Python
        if lang in ["python", "py", "python3"]:
            metrics["python_checked"] += 1
            try:
                # Análise de árvore sintática (AST)
                ast.parse(code_clean)
                # Compilação para bytecode sem executar
                compile(code_clean, f"<star_sandbox_py_{idx}>", "exec")
            except SyntaxError as syn_err:
                reasons.append(
                    f"Erro de sintaxe Python no bloco {idx} (linha {syn_err.lineno}): {syn_err.msg}"
                )
            except Exception as e:
                reasons.append(f"Erro na análise do bloco Python {idx}: {str(e)}")

        # 2. Verificação Sintática de Regras YARA
        elif lang in ["yara", "yar"]:
            metrics["yara_checked"] += 1
            # Validação estrutural de YARA
            if not re.search(r"\brule\s+[a-zA-Z_][a-zA-Z0-9_]*", code_clean):
                reasons.append(
                    f"Bloco YARA {idx} não contém cabeçalho de regra válido: 'rule <identifier>'."
                )
            if "condition:" not in code_clean:
                reasons.append(f"Bloco YARA {idx} ausenta a seção obrigatória 'condition:'.")

            # Balanceamento de chaves {}
            open_braces = code_clean.count("{")
            close_braces = code_clean.count("}")
            if open_braces != close_braces or open_braces == 0:
                reasons.append(
                    f"Bloco YARA {idx} possui chaves desbalanceadas: {open_braces} '{{' vs {close_braces} '}}'."
                )

        # 3. Verificação de Regras Sigma e YAML
        elif lang in ["yaml", "yml", "sigma"]:
            if lang == "sigma":
                metrics["sigma_checked"] += 1
            else:
                metrics["yaml_checked"] += 1

            if YAML_AVAILABLE:
                try:
                    parsed_yaml = yaml.safe_load(code_clean)
                    if not isinstance(parsed_yaml, (dict, list)) and parsed_yaml is not None:
                        reasons.append(f"Bloco {lang} {idx} gerou estrutura YAML inválida (esperado dict ou list).")
                    elif lang == "sigma" or ("logsource" in code_clean and "detection" in code_clean):
                        # Validação de campos essenciais da especificação Sigma
                        if isinstance(parsed_yaml, dict):
                            missing_sigma_keys = []
                            for k in ["title", "logsource", "detection"]:
                                if k not in parsed_yaml:
                                    missing_sigma_keys.append(k)
                            if missing_sigma_keys:
                                reasons.append(
                                    f"Regra Sigma {idx} ausenta chaves obrigatórias da especificação: {missing_sigma_keys}."
                                )
                except Exception as y_err:
                    reasons.append(f"Erro de sintaxe YAML no bloco {idx}: {str(y_err)}")
            else:
                # Fallback se PyYAML não estiver disponível: validação básica de indentação
                if "\t" in code_clean:
                    reasons.append(f"Bloco YAML {idx} contém tabulações proibidas na especificação.")

    passed = len(reasons) == 0
    return passed, reasons, metrics


# ==============================================================================
# Classe Orquestradora: STaRVerifier
# ==============================================================================

class STaRVerifier:
    """
    Orquestrador dos três verificadores determinísticos (RLVR / STaR).
    Calcula pontuação ponderada e aplica Rejection & Pruning rigoroso.
    """

    def __init__(
        self,
        min_think_chars: int = 80,
        min_think_words: int = 15,
        min_final_chars: int = 20,
        strict_format: bool = True,
        strict_ground_truth: bool = True,
        strict_code_syntax: bool = True,
        check_inside_think_code: bool = False,
        include_prm: bool = False,
        w_prm: float = 0.0,
        enable_prm: Optional[bool] = None,
    ):
        self.min_think_chars = min_think_chars
        self.min_think_words = min_think_words
        self.min_final_chars = min_final_chars
        self.strict_format = strict_format
        self.strict_ground_truth = strict_ground_truth
        self.strict_code_syntax = strict_code_syntax
        self.check_inside_think_code = check_inside_think_code
        self.include_prm = enable_prm if enable_prm is not None else include_prm
        self.enable_prm = self.include_prm
        self.w_prm = w_prm

    @staticmethod
    def check_refusals(text: str) -> List[str]:
        """Detecta recusas conversacionais e epistêmicas no texto."""
        return check_refusals(text)

    def score_prm_steps(self, candidate: str, prompt: str = "") -> Dict[str, Any]:
        """Calcula a avaliação detalhada do Process Reward Model nos passos de raciocínio."""
        return verify_prm_steps(candidate, prompt)

    def verify_candidate(
        self,
        problem: Dict[str, Any],
        candidate: str
    ) -> VerificationResult:
        """
        Executa os 3 verificadores em uma resposta candidata e calcula a pontuação total.
        """
        all_reasons: List[str] = []
        metrics: Dict[str, Any] = {}

        # 1. Formato e CoT
        fmt_ok, fmt_reasons, fmt_metrics = verify_format_integrity(
            candidate,
            min_think_chars=self.min_think_chars,
            min_think_words=self.min_think_words,
            min_final_chars=self.min_final_chars,
        )
        all_reasons.extend(fmt_reasons)
        metrics["format"] = fmt_metrics

        # 2. Ground-Truth Exact Match
        gt_ok, gt_reasons, gt_metrics = verify_ground_truth(candidate, problem)
        all_reasons.extend(gt_reasons)
        metrics["ground_truth"] = gt_metrics

        # 3. Sintaxe de Código
        code_ok, code_reasons, code_metrics = verify_code_syntax(
            candidate,
            check_inside_think=self.check_inside_think_code,
        )
        all_reasons.extend(code_reasons)
        metrics["code_syntax"] = code_metrics

        # 4. Step-Level PRM (v1.4)
        prm_metrics = verify_prm_steps(candidate)
        metrics["prm"] = prm_metrics

        # Decisão de aprovação estrita
        passed = True
        if self.strict_format and not fmt_ok:
            passed = False
        if self.strict_ground_truth and not gt_ok:
            passed = False
        if self.strict_code_syntax and not code_ok:
            passed = False

        if self.include_prm or getattr(self, "enable_prm", False):
            # Integração ponderada v1.4 com Process Reward Model
            prm_score = float(prm_metrics.get("prm_score", 0.0))
            fmt_score = 1.0 if fmt_ok else 0.0
            gt_score = 1.0 if gt_ok else 0.0
            cd_score = 1.0 if code_ok else 0.0
            score = compute_v14_composite_score(
                format_score=fmt_score,
                gt_score=gt_score,
                prm_score=prm_score,
                code_score=cd_score,
                w_prm=self.w_prm if self.w_prm > 0 else 0.25,
            )
        else:
            # Cálculo clássico retrocompatível (0.0 a 1.0)
            score = 0.0

            if gt_ok:
                score += 0.40

            if fmt_ok:
                score += 0.20

            # Bônus de cobertura de estágios analíticos (0 a 0.20)
            stages_cov = fmt_metrics.get("stages_coverage", 0.0)
            score += float(stages_cov) * 0.20

            # Código e estrutura
            if code_ok:
                score += 0.10
            if fmt_metrics.get("structure_markers_found", 0) > 0:
                score += 0.10

        # Se falhar estritamente, penaliza a pontuação
        if not passed:
            score = max(0.0, score * 0.25)

        return VerificationResult(
            passed=passed,
            score=round(score, 4),
            format_ok=fmt_ok,
            ground_truth_ok=gt_ok,
            code_syntax_ok=code_ok,
            reasons=all_reasons,
            metrics=metrics,
        )

    def verify_and_score(
        self,
        problem: Dict[str, Any],
        candidate: str,
        use_prm: Optional[bool] = None,
    ) -> VerificationResult:
        """Executa verificação e pontuação com suporte explícito a PRM v1.4."""
        prev_include = self.include_prm
        prev_enable = getattr(self, "enable_prm", False)
        if use_prm is not None:
            self.include_prm = use_prm
            self.enable_prm = use_prm
        try:
            return self.verify_candidate(problem, candidate)
        finally:
            self.include_prm = prev_include
            self.enable_prm = prev_enable

    def filter_and_rank_candidates(
        self,
        problem: Dict[str, Any],
        candidates: List[str],
        use_prm: Optional[bool] = None,
    ) -> List[Tuple[str, VerificationResult]]:
        """
        Aplica Rejection & Pruning:
        - Descarta candidatos reprovados (passed == False).
        - Ordena os candidatos aprovados pela pontuação de qualidade (decrescente).
        """
        prev_include = self.include_prm
        prev_enable = getattr(self, "enable_prm", False)
        if use_prm is not None:
            self.include_prm = use_prm
            self.enable_prm = use_prm
        try:
            verified = []
            for cand in candidates:
                res = self.verify_candidate(problem, cand)
                if res.passed:
                    verified.append((cand, res))

            # Ordenar por pontuação decrescente
            verified.sort(key=lambda item: item[1].score, reverse=True)
            return verified
        finally:
            self.include_prm = prev_include
            self.enable_prm = prev_enable

    def rank_candidates(
        self,
        problem: Dict[str, Any],
        candidates: List[str],
        use_prm: Optional[bool] = None,
    ) -> List[Tuple[str, VerificationResult]]:
        """Alias para filter_and_rank_candidates."""
        return self.filter_and_rank_candidates(problem, candidates, use_prm=use_prm)


# ==============================================================================
# Interface CLI para Testes e Auditoria
# ==============================================================================

def run_cli():
    parser = argparse.ArgumentParser(
        description="CyberRott 1.1 - Deterministic Verification Engine (STaR / RLVR)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Caminho do arquivo JSONL contendo amostras ou candidatos para verificação."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Caminho opcional do arquivo JSONL para salvar as amostras verificadas com sucesso."
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Modo auditoria: valida amostras completas do dataset (usando resposta do assistant)."
    )
    parser.add_argument(
        "--min_think_chars",
        type=int,
        default=80,
        help="Mínimo de caracteres exigidos dentro da tag <think> (padrão: 80)."
    )
    parser.add_argument(
        "--strict_code",
        action="store_true",
        default=True,
        help="Rejeitar sumariamente se houver erro de sintaxe em código gerado (padrão: True)."
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[Erro] Arquivo de entrada não encontrado: {args.input}")
        return

    verifier = STaRVerifier(
        min_think_chars=args.min_think_chars,
        strict_code_syntax=args.strict_code,
    )

    total = 0
    passed_count = 0
    fmt_pass = 0
    gt_pass = 0
    code_pass = 0
    scores = []

    out_file = open(args.output, "w", encoding="utf-8") if args.output else None

    print("=" * 70)
    print("🔍 Auditoria Determinística STaR / RLVR - CyberRott 1.1")
    print(f"Arquivo de Entrada:  {args.input}")
    print(f"Modo Auditoria:      {args.audit}")
    print(f"Min Chars <think>:   {args.min_think_chars}")
    print("=" * 70)

    with open(args.input, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except Exception as e:
                print(f"[Linha {idx}] Erro de parsing JSON: {e}")
                continue

            total += 1

            # Extração de candidato e metadados
            candidate = ""
            if args.audit or "messages" in data:
                # Extrai resposta do assistente como candidato
                messages = data.get("messages", [])
                for msg in messages:
                    if msg.get("role") == "assistant":
                        candidate = msg.get("content", "")
                        break
            elif "candidate" in data:
                candidate = data.get("candidate", "")
            elif "generated_text" in data:
                candidate = data.get("generated_text", "")

            result = verifier.verify_candidate(data, candidate)
            scores.append(result.score)

            if result.format_ok:
                fmt_pass += 1
            if result.ground_truth_ok:
                gt_pass += 1
            if result.code_syntax_ok:
                code_pass += 1
            if result.passed:
                passed_count += 1
                if out_file:
                    out_record = dict(data)
                    out_record["star_verification"] = result.to_dict()
                    out_file.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    if out_file:
        out_file.close()

    # Relatório estatístico
    mean_score = (sum(scores) / len(scores)) if scores else 0.0
    pass_rate = (passed_count / total * 100) if total else 0.0

    print("\n" + "=" * 70)
    print("📊 RESULTADOS DA AUDITORIA DETERMINÍSTICA:")
    print(f"Total de Amostras Avaliadas:    {total}")
    print(f"Format Integrity Pass Rate:     {fmt_pass}/{total} ({fmt_pass/max(total, 1)*100:.1f}%)")
    print(f"Ground-Truth Match Pass Rate:   {gt_pass}/{total} ({gt_pass/max(total, 1)*100:.1f}%)")
    print(f"Code Syntax Sandbox Pass Rate:  {code_pass}/{total} ({code_pass/max(total, 1)*100:.1f}%)")
    print(f"Taxa Global de Aprovação STaR:  {passed_count}/{total} ({pass_rate:.1f}%)")
    print(f"Pontuação Média de Qualidade:   {mean_score:.4f}")
    if args.output:
        print(f"Amostras aprovadas salvas em:   {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    run_cli()
