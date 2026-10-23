---
language:
- en
- pt
license: mit
base_model: microsoft/Phi-4-mini-instruct
tags:
- cybersecurity
- reinforcement-learning
- grpo
- deepseek-r1
- reasoning
- cot
- chain-of-thought
- slm
- red-team
- blue-team
- vulnerability-analysis
- appsec
- cve
- gguf
- ollama
datasets:
- nvidia/Nemotron-Research-GooseReason-0.7M
- m-a-p/CodeFeedback-Filtered-Instruction
- open-r1/codeforces-cots
- bespokelabs/Bespoke-Stratos-17k
- CyberNative/Code_Vulnerability_Security_DPO
- AlicanKiraz0/Cybersecurity-Dataset-v1
- cais/wmdp
pipeline_tag: text-generation
model-index:
- name: CyberRott-v1.2-GRPO
  results:
  - task:
      type: text-generation
      name: Cybersecurity Question Answering & Vulnerability Auditing
    dataset:
      name: WMDP-Cyber
      type: cais/wmdp
      args: wmdp-cyber
    metrics:
    - name: WMDP-Cyber Accuracy
      type: acc
      value: 43.18
---

# 🛡️ CyberRott v1.2 — Reasoning-Driven Cybersecurity SLM (DeepSeek-R1 Style GRPO)

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Base Model: Phi-4-mini](https://img.shields.io/badge/Base%20Model-Phi--4--mini--instruct%20(3.8B)-792ee5.svg?style=for-the-badge)](https://huggingface.co/microsoft/Phi-4-mini-instruct)
[![Post-Training: GRPO](https://img.shields.io/badge/Post--Training-DeepSeek--R1%20GRPO-ff6f00.svg?style=for-the-badge)](https://github.com/huggingface/trl)
[![Format: GGUF Q4_K_M](https://img.shields.io/badge/Format-GGUF%20Q4__K__M%20(2.4GB)-00bcd4.svg?style=for-the-badge)](https://github.com/ggerganov/llama.cpp)
[![Framework: TRL + PEFT](https://img.shields.io/badge/Framework-TRL%20%2B%20PEFT%20(LoRA)-orange.svg?style=for-the-badge)](https://github.com/huggingface/peft)

**A 3.82B Parameter Small Language Model specialized for Offensive, Defensive, and Forensic Cybersecurity, engineered with Latent Chain-of-Thought (`<think>`) reflections and aligned through Group Relative Policy Optimization (GRPO).**

</div>

---

## 📌 Model Overview

**CyberRott v1.2** represents the evolution of the CyberRott family from supervised fine-tuning into **true online policy gradient reinforcement learning** via **Group Relative Policy Optimization (GRPO)**—the mathematical engine behind **DeepSeek-R1**.

Operated atop **Microsoft Phi-4-mini-instruct** (3.82B dense parameters), CyberRott v1.2 enforces an internal deliberation phase enclosed within `<think>...</think>` tokens, autonomously examining:
1. **Surface Analysis & Attack Vector Enumeration**
2. **Untrusted Entry Points & Taint Analysis Propagation Paths**
3. **Dangerous Sinks & Flawed Boundaries**
4. **Existing Sanitization & Defensive Mitigations**
5. **Exact CWE & MITRE ATT&CK Classification**
6. **Defensive Source Remediation with Boundary Checks**

---

## 🚀 Quickstart: Local Execution with Ollama

Run CyberRott v1.2 locally in 1 command using Ollama (no manual weight conversion needed):

```bash
ollama run hf.co/johnknevescarrijo/CyberRott-v1.2-GRPO
```

### Prompt Example:
```text
Analise a função C abaixo para vulnerabilidades de memória:

void parse_packet(const char *payload, int len) {
    char buf[64];
    if (len > 0) strcpy(buf, payload);
    process(buf);
}
```

### CyberRott v1.2 Generated Response:
```text
<think>
A função parse_packet recebe payload e comprimento, copiando para um buffer local de 64 bytes sem verificar se o tamanho excede o limite. Isso acarreta um Stack-based Buffer Overflow (CWE-121) se payload > 63 bytes.
Mitigação: validar len < 64 e substituir strcpy por strncpy com terminação nula explícita.
</think>

Vulnerabilidade Identificada: Buffer Overflow na Pilha (CWE-121)
Impacto: Execução arbitrária de código (Crítico)

Patch Corretivo Defensivo:
```c
#define BUF_SIZE 64
void parse_packet(const char *payload, int len) {
    char buf[BUF_SIZE];
    if (len > 0 && len < BUF_SIZE) {
        strncpy(buf, payload, BUF_SIZE - 1);
        buf[BUF_SIZE - 1] = '\0';
    }
    process(buf);
}
```
```

---

## 📚 Curated Training & Alignment Datasets

CyberRott v1.2 utilizes an open-source, Zero-API-Cost curated dataset corpus:

| Dataset | Hugging Face Repository | Primary Domain & Application |
| :--- | :--- | :--- |
| **NVIDIA GooseReason-0.7M** | [`nvidia/Nemotron-Research-GooseReason-0.7M`](https://huggingface.co/datasets/nvidia/Nemotron-Research-GooseReason-0.7M) | Deep analytical reasoning, math, and structured code logic |
| **CodeFeedback Filtered** | [`m-a-p/CodeFeedback-Filtered-Instruction`](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) | Code review, memory corruption analysis & safe refactoring |
| **Open-R1 CoTs** | [`open-r1/codeforces-cots`](https://huggingface.co/datasets/open-r1/codeforces-cots) | Long Chain-of-Thought deliberation trajectories (DeepSeek-R1 style) |
| **Bespoke Stratos 17k** | [`bespokelabs/Bespoke-Stratos-17k`](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) | High-density multi-step reasoning trajectories |
| **CyberNative Security DPO** | [`CyberNative/Code_Vulnerability_Security_DPO`](https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO) | Vulnerable vs. secure code pairs with exact CWE taxonomy |
| **AlicanKiraz0 Cyber v1** | [`AlicanKiraz0/Cybersecurity-Dataset-v1`](https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-v1) | Red/Blue Team operations, MITRE ATT&CK tactics & digital forensics |
| **WMDP-Cyber** | [`cais/wmdp`](https://huggingface.co/datasets/cais/wmdp) | Official academic benchmark for high-hazard cybersecurity capabilities |

---

## ⚙️ Technical Specifications & Training Details

| Parameter | Specification | Technical Rationale |
| :--- | :--- | :--- |
| **Base Architecture** | `microsoft/Phi-4-mini-instruct` | 3.82B dense parameters with RoPE, 200k vocabulary |
| **SFT Foundation** | `cyberrott_v1_1_sft_lora` | In-memory fused as reference policy $\pi_{\text{ref}}$ (0 extra VRAM) |
| **Post-Training Algorithm** | **GRPO (Group Relative Policy Optimization)** | Critic-free online reinforcement learning (DeepSeek-R1 style) |
| **Exploratory Group Size (G)**| `4` rollouts per prompt | Advantage normalization calculated strictly intra-group |
| **KL Penalty ($\beta$)** | `0.04` | Regularizes policy against drifting away from SFT domain fluency |
| **PEFT Method** | LoRA (`r=32`, `\alpha=64`, `all-linear`) | Trainable adapter over `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj` |
| **Numerical Precision** | `bfloat16` | Dynamic exponent range preventing gradient underflow |
| **Hardware & Runtime** | 1x NVIDIA A100-SXM4 (40GB) | **46.8 minutes** (`2,812s`) total convergence time |

---

## 🎯 Verifiable Reward Suite (`reward_funcs`)

CyberRott v1.2 replaces heuristic reward models with **3 deterministic, verifiable reward functions**:

1. **CoT Integrity Reward (`cot_integrity_reward` — Weight: 0.30)**:
   - Validates balanced `<think>...</think>` latent reflection blocks.
   - Evaluates depth across the 7 analytical checkpoints.
   - Imposes immediate `0.0` score penalty for conversational refusals (*"As an AI..."*, *"I cannot..."*).
2. **Technical Accuracy Reward (`technical_accuracy_reward` — Weight: 0.50)**:
   - Verifies exact multiple-choice keys (`A`, `B`, `C`, `D`).
   - Validates Common Weakness Enumeration identifiers (e.g. `CWE-121`, `CWE-89`).
   - Validates MITRE ATT&CK technique IDs (e.g. `T1059`, `T1190`).
3. **Sandbox Code Reward (`sandbox_code_reward` — Weight: 0.20)**:
   - Python AST static compilation check (`compile(source, mode="exec")`).
   - Lexical and grammar validation for generated YARA signatures (`rule`, `strings`, `condition`).
   - Schema and syntax validation for Sigma detection rules.

---

## 👨‍💻 Authors & Citation

- **Lead Architect and Developer**: [John Carrijo](https://github.com/johnknevescarrijo)
- **Source Code Repository**: [https://github.com/johnknevescarrijo/CyberRott](https://github.com/johnknevescarrijo/CyberRott)
- **Base Architecture**: Microsoft Research (`Phi-4-mini-instruct`)
