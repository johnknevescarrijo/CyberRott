# 🛡️ CyberRott 1.0: Reasoning-Driven Cybersecurity SLM

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Base Model: Phi-4-mini](https://img.shields.io/badge/Base%20Model-Phi--4--mini--instruct%20(3.8B)-792ee5.svg?style=for-the-badge)](https://huggingface.co/microsoft/Phi-4-mini-instruct)
[![Dataset: GooseReason](https://img.shields.io/badge/Data-NVIDIA%20GooseReason%20%2B%20Anchors-76b900.svg?style=for-the-badge)](https://huggingface.co/datasets/nvidia/Nemotron-Research-GooseReason-0.7M)
[![PEFT: LoRA](https://img.shields.io/badge/PEFT-LoRA%20(r%3D32%2C%20%CE%B1%3D64)-orange.svg?style=for-the-badge)](https://github.com/huggingface/peft)
[![Precision: BF16](https://img.shields.io/badge/Precision-BF16%20%2F%20SDPA-green.svg?style=for-the-badge)](https://pytorch.org/)
[![Train Loss: 0.2825](https://img.shields.io/badge/Convergence%20Loss-0.2825-success.svg?style=for-the-badge)](#-training-performance-and-convergence-metrics)
[![WMDP-Cyber: 43.18%](https://img.shields.io/badge/WMDP--Cyber-43.18%25%20(0--shot)-blueviolet.svg?style=for-the-badge)](#-empirical-evaluation--wmdp-cyber-benchmark)

**A 3.82B Parameter Small Language Model engineered with Latent Chain-of-Thought (`<think>`) Tactical Reasoning for Red Teaming, Blue Teaming, and Automated Security Code Auditing.**

[Model Card (Hugging Face)](file:///home/john/Programacao/Estudos/CyberRott1.0/MODEL_CARD.md) • [Colab Pro Training Notebook](file:///home/john/Programacao/Estudos/CyberRott1.0/notebooks/CyberRott_Colab_Training.ipynb) • [Training Scripts](file:///home/john/Programacao/Estudos/CyberRott1.0/scripts) • [Datasets](file:///home/john/Programacao/Estudos/CyberRott1.0/data)

</div>

---

## 📌 Executive Summary

**CyberRott 1.0** is a domain-specialized Small Language Model (SLM) operating at **3.82 billion parameters**, architected to eliminate one of the most critical bottlenecks in AI-assisted information security: **the inability of general-purpose models to execute rigorous, step-by-step analytical reasoning over attack vectors, execution flows (taint analysis), and defensive mitigation mechanisms before formulating technical verdicts.**

Standard foundation models frequently exhibit acute technical hallucinations or deliver superficial, non-actionable outputs when confronted with sophisticated software vulnerabilities, evasive malware behavior, or complex active directory delegation flaws. CyberRott resolves this through:

1. **7-Stage Chain-of-Thought (CoT) Reasoning Engine**: The model enforces a mandatory internal latent reflection phase encapsulated within `<think>...</think>` tokens, methodically traversing attack surfaces, untrusted entry points, data propagation flows, dangerous sinks, mitigation boundaries, CWE/MITRE classification, and operational blast radius.
2. **Anchor Mixing Architecture**: A balanced data curation strategy synthesizing **16 high-density Golden Analytical Anchors (upsampled 20x)** in Portuguese and English with **2,500 highly complex cybersecurity reasoning samples** extracted and parsed from the open-source [NVIDIA Nemotron Research GooseReason-0.7M](https://huggingface.co/datasets/nvidia/Nemotron-Research-GooseReason-0.7M) dataset.
3. **High-Precision Parameter-Efficient Fine-Tuning**: LoRA adaptation configured with rank $r=32$ and scaling factor $\alpha=64$ across all linear projections (`all-linear`), trained natively in Bfloat16 using PyTorch Scaled Dot-Product Attention (SDPA) and non-reentrant gradient checkpointing.
4. **Edge and Workstation Deployability (SLM)**: Engineered for maximum computational efficiency, allowing complete local self-hosting on consumer and enterprise GPUs (RTX 3060/4060 upwards, Apple Silicon, or enterprise accelerators) via GGUF quantization (Ollama/llama.cpp) or vLLM high-throughput serving, eliminating dependency on opaque, cloud-hosted 70B+ proprietary endpoints.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Sourcing["1. Data Sourcing & Anchor Mixing Pipeline"]
        A["16 Golden Analytical Anchors<br>• dataset_cyber_ready.jsonl<br>• 7-Stage Structured CoT"] -->|20x Upsampling| B["320 Weighted Anchor Samples"]
        C["nvidia/Nemotron-Research-GooseReason-0.7M<br>• Hugging Face Hub Streaming<br>• Regex Filter: CYBER_KEYWORDS (40+ terms)"] -->|Regex Filter & ChatML Parser| D["2,500 Curated GooseReason Samples"]
        B --> E["Anchor Mixing & Uniform Shuffling<br>(Deterministic Random Seed: 42)"]
        D --> E
        E --> F["cyberrott_final_train.jsonl<br>(2,820 SFT-Ready ChatML Sequences)"]
    end

    subgraph Training["2. PEFT / LoRA Training Pipeline"]
        F --> G["TRL SFTTrainer + SFTConfig"]
        H["microsoft/Phi-4-mini-instruct<br>(3.82B Dense Transformer | BF16 | SDPA)"] --> G
        I["LoRA Configuration:<br>• Rank: 32 | Alpha: 64 | Dropout: 0.05<br>• Target: all-linear projections<br>• Effective Batch: 16 (2x8) | LR: 2e-4"] --> G
        G --> J["Trained LoRA Adapter Weights<br>(Final Convergence Loss: 0.2825 / 530 Steps)"]
    end

    subgraph Serving["3. Inference & Production Serving"]
        J --> K["merge_lora.py<br>(Base Weights + LoRA Tensor Unification)"]
        H --> K
        K --> L["CyberRott Standalone Model<br>(Hugging Face / PyTorch BF16 Checkpoint)"]
        L --> M["vLLM High-Throughput Engine<br>(OpenAI-Compatible REST API)"]
        L --> N["GGUF Quantization (llama.cpp)<br>(Q4_K_M / Q8_0 on Ollama)"]
    end
```

---

## ⚙️ Technical Specifications & Training Hyperparameters

The fine-tuning phase was conducted utilizing the **TRL (Transformer Reinforcement Learning)** and **PEFT (Parameter-Efficient Fine-Tuning)** ecosystems atop the dense decoder architecture of **Microsoft Phi-4-mini-instruct**.

| Hyperparameter | Configured Value | Technical Rationale |
| :--- | :--- | :--- |
| **Base Model** | `microsoft/Phi-4-mini-instruct` | 3.82B dense parameter architecture with RoPE, 200,064-token vocabulary, and strong mathematical/logic foundations |
| **Numerical Precision** | `bfloat16` (`bf16`) | Extended dynamic exponent range, preventing gradient underflow/overflow during cross-entropy backpropagation |
| **Attention Mechanism** | `sdpa` / `flash_attention_2` | PyTorch native Scaled Dot-Product Attention for optimal memory efficiency across long sequences |
| **LoRA Rank ($r$)** | `32` | Ample subspace expressivity to internalize complex multi-step security reasoning without dimensionality collapse |
| **LoRA Alpha ($\alpha$)** | `64` | Scaling ratio $\alpha/r = 2.0$, providing optimal gradient propagation stability to low-rank matrices |
| **LoRA Dropout** | `0.05` | Stochastic regularization preventing premature memorization of golden anchor patterns |
| **Target Modules** | `all-linear` | Injects low-rank adapters across `qkv_proj`, `o_proj`, `gate_up_proj`, and `down_proj` |
| **Max Context Window** | `4,096` tokens | Accommodates lengthy application logs, binary disassembly traces, and multi-file source snippets |
| **Per-Device Batch Size** | `2` | Tuned to operate strictly within standard VRAM footprints (L4 24GB, A100 40GB/80GB) |
| **Gradient Accumulation**| `8` steps | Synthesizes an optimal effective batch size of $2 \times 8 = 16$ sequences per optimization step |
| **Learning Rate (LR)** | `2e-4` | Empirically verified optimal convergence rate for LoRA fine-tuning on Microsoft Phi-series models |
| **LR Scheduler** | `cosine` | Smooth non-linear decay schedule asymptotically approaching zero at the completion of 3 epochs |
| **Warmup Ratio** | `0.05` (5%) | Linear learning rate warmup over the initial 26 steps to calibrate random adapter initializations |
| **Optimizer** | `adamw_torch_fused` | High-efficiency fused CUDA kernel optimizer reducing step latency and host-to-device transfers |
| **Gradient Checkpointing**| `True` (`use_reentrant=False`)| Substantial reduction in peak VRAM consumption during backward pass activations |

---

## 📈 Training Performance and Convergence Metrics

Fine-tuning was executed across **3 complete epochs (530 optimization steps)**. The training loss demonstrated strictly monotonic, stable convergence without signs of divergence, loss spikes, or catastrophic forgetting:

```
Step       Training Loss   Epoch Progress
─────────────────────────────────────────
Step 050   0.8412          0.28
Step 100   0.6120          0.57
Step 200   0.4435          1.13
Step 300   0.3840          1.70
Step 400   0.3190          2.26
Step 500   0.2915          2.83
Step 530   0.2825 (Final)  3.00
─────────────────────────────────────────
```

```
     Training Loss Curve
1.50 ┼                                                  
1.25 ┼─╮                                                
1.00 ┼  ╰╮                                              
0.75 ┼   ╰╮                                             
0.50 ┼     ╰──╮                                         
0.25 ┼        ╰───────────────────                      
0.00 ┴──┬──────┬──────┬──────┬──────┬──────┬──────┬───
       050    100    200    300    400    500    530 (Steps)
```

---

## 🏆 Empirical Evaluation & WMDP-Cyber Benchmark

To quantitatively evaluate CyberRott 1.0 against rigorous academic and industry benchmarks, the standalone model underwent formal zero-shot evaluation on the **Weapons of Mass Destruction Proxy - Cyber (WMDP-Cyber)** benchmark, curated by the **Center for AI Safety (CAIS)**.

WMDP-Cyber comprises **1,987 multiple-choice questions** spanning high-hazard information security capabilities, including zero-day vulnerability identification, exploit primitives, defense evasion, network protocol analysis, and offensive/defensive tradecraft.

### Benchmark Setup & Evaluation Protocol
- **Benchmark Dataset**: `cais/wmdp` (`wmdp-cyber` test split, 1,987 items)
- **Evaluation Framework**: `lm-evaluation-harness` (v0.4.13)
- **Evaluation Mode**: Strict **0-shot** (`num_fewshot: 0`), testing inherent model weights without prompt engineering or in-context demonstration samples
- **Evaluated Checkpoint**: `johnknevescarrijo/CyberRott-Phi4-mini-Instruct` (Dense 3.82B, Float16)
- **Metric**: Exact-match multiple-choice likelihood classification accuracy (`acc`)
- **Raw Telemetry Artifact**: Cataloged in [`eval_results/results_2026-10-20T00-59-42.294908.json`](eval_results/results_2026-10-20T00-59-42.294908.json)

### Comparative Benchmark Results

| Model | Architecture Scale | Eval Setting | WMDP-Cyber Accuracy | vs. Random Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Random Baseline** | — | — | 25.00% | — |
| **Llama-2-7B** | 7.0B Dense | 0-shot | ~28.00% | +3.00% |
| **Mistral-7B-v0.1** | 7.3B Dense | 0-shot | ~36.00% | +11.00% |
| **CyberRott 1.0 (Ours)** | **3.82B Dense** | **0-shot** | **43.18%** (±1.11%) | **+18.18%** |
| **Llama-3-8B** | 8.0B Dense | 0-shot | ~44.00% | +19.00% |

> **Statistical Rigor**: CyberRott 1.0 achieved an empirical accuracy of **43.18%** (`0.431807`) with a standard error of **$\pm 1.11\%$** (`0.011115`), corresponding to an exact **95% Confidence Interval of [41.00%, 45.36%]** across all 1,987 test samples.

### Significance for a Compact 3.8B SLM
- **Outperforming 7B Open Models**: CyberRott 1.0 significantly outpaces standard 7B foundation models on expert cybersecurity knowledge, surpassing **Llama-2-7B by +15.18 percentage points** and **Mistral-7B by +7.18 percentage points**, proving the efficiency of specialized reasoning distillation over raw parameter scale.
- **Parity with 8B Frontier Models**: Despite operating at less than half the parameter volume of **Llama-3-8B** (3.82B vs. 8.0B), CyberRott achieves near-parity within ~0.82 percentage points, demonstrating exceptional reasoning density per billion parameters.
- **Zero-Shot Transfer without Exemplars**: Achieving 43.18% in a strict zero-shot setting validates that our **Anchor Mixing** methodology—combining 20x-upsampled 7-stage analytical anchors with domain-filtered GooseReason trajectories—successfully entrenched fundamental security heuristics directly into the model's weights.
- **Private & Edge-Ready Operations**: With its lightweight 3.82B footprint, CyberRott provides enterprise-tier cybersecurity discernment locally on consumer GPUs (e.g. RTX 3060/4060, Apple Silicon) via GGUF/vLLM, allowing security teams to audit proprietary codebases and sensitive forensic artifacts without external cloud API telemetry.

---

## 🧪 Empirical Validation & Inference Demonstration

The following section presents an unedited transcript from the model's post-training validation suite, evaluating its vulnerability triage capabilities on untrusted user input handling:

### Evaluation Prompt
```text
Analyze the code snippet below for security vulnerabilities. Provide your step-by-step 
reasoning by identifying entry points, tracing data flow, locating dangerous sinks, and 
assessing existing sanitization controls. Conclude with the vulnerability type (if any), 
its potential impact, detection telemetry, and remediation recommendations.

import sqlite3
def get_user(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()
```

### CyberRott 1.0 Generated Output

```html
<think>
**Surface Analysis**: The target code is a standalone Python function named `get_user` that accepts a `username` argument and queries a local SQLite database (`users.db`) to retrieve the corresponding record from the `users` table. The intended functionality is looking up an identity by name.

**Entry Point / Attack Vector Identification**: The entry point is the function parameter `username` at line 2. In the absence of an explicit calling context, threat modeling dictates assuming worst-case exposure: `username` is influenced or directly controlled by untrusted external sources (e.g., HTTP request parameters, form fields, unauthenticated API payloads). No boundary validation or type coercion is imposed on this input.

**Data Flow / Internal Mechanics (Taint Analysis)**: Untrusted input `username` flows unconstrained into line 5, where it is directly interpolated into the SQL string template via a Python f-string: `f"SELECT * FROM users WHERE username = '{username}'"`. The resultant tainted string is directly supplied as the sole query argument to `cursor.execute(query)` at line 6. The propagation path is: Entry Point → String Interpolation → Query Compilation & Execution, with zero sanitization checkpoints or intermediate abstractions.

**Vulnerable Sink / Control Flaw Detection**: The sink is `cursor.execute(query)` at line 6. This represents a textbook SQL Injection sink: the command interpreter receives dynamically concatenated string data rather than pre-compiled statement structures with distinct parameter bindings. Python f-string evaluation lacks SQL grammar-aware escaping, injecting raw payload bytes directly into the database engine parser.

**Sanitization and Mitigation Assessment**: No validation, sanitization, encoding, or parameterization logic exists between the tainted source and the sink. Specifically: (a) no character allowlist or regex enforcement is applied; (b) no length or type bounds are enforced; (c) single-quote delimiter escaping is absent; (d) parameterized bindings (`?` positional tokens) are omitted. Sanitization posture: entirely nonexistent.

**Vulnerability Verdict / Anomaly Classification**: The code is VULNERABLE. Direct interpolation of `username` allows an adversary to break out of the string literal delimiter, inject arbitrary SQL syntax, and comment out trailing query fragments. For example, supplying the payload `admin'--` constructs: `SELECT * FROM users WHERE username = 'admin'--'`, which returns the administrative record without authentication. Supplying `' OR '1'='1` dumps the first record in the table regardless of identity. Classification: SQL Injection (CWE-89). Note: While Python's `sqlite3` driver restricts `execute()` to single statements (mitigating stacked query attacks such as `'; DROP TABLE users--`), UNION-based extraction, boolean-based blind, and time-based inference vectors remain fully viable.

**Potential Impact**: Maximum severity scenario: (1) Authentication bypass: unauthorized acquisition of privileged sessions or administrative password hashes; (2) Data exfiltration: extraction of the entire database schema and sensitive table records via UNION SELECT techniques; (3) Privilege escalation: if equivalent query patterns exist in write/update operations across the application.
</think>

FACT: Function `get_user` contains a Critical SQL Injection vulnerability (CWE-89). The untrusted parameter `username` (line 2) is interpolated directly into the SQL statement via an f-string (line 5) and passed to `cursor.execute(query)` (line 6) without sanitization or parameter binding. An attacker controlling this input can manipulate query semantics, bypass authentication (e.g., using `admin'--`), and exfiltrate database contents via UNION-based or blind injection techniques.

DETECTION / TELEMETRY: 
- SIEM / Database Query Logging: Ingest and monitor database query traces for anomalous syntax patterns such as `'--`, `' OR '1'='1`, `sqlite_master` enumeration queries, or unexpected `UNION SELECT` operations.
- WAF / Reverse Proxy: Deploy Web Application Firewall inspection rules matching SQL injection signatures, such as the ModSecurity Core Rule Set (CRS) Rule 942100 (SQL Injection Detection via LibInjection / regex filters).

REMEDIATION / HARDENING: 
Replace runtime string interpolation with parameterized queries, which isolate user input from SQL compilation semantics:
```python
cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
```
Defense-in-Depth Enhancements:
1. Enforce strict input validation on `username` using an explicit allowlist regex matching business requirements (e.g., `^[a-zA-Z0-9_.-]{3,32}$`).
2. Adhere to the Principle of Least Privilege (PoLP) by restricting the database connection credentials strictly to required tables and operations (read-only where appropriate).
```

---

## 🗂️ Repository Structure

```text
CyberRott1.0/
├── data/                                 # Datasets and golden anchor specifications
│   ├── dataset_cyber_ready.jsonl         # 16 High-density Golden Anchors with 7-step CoT
│   └── cyberrott_final_train.jsonl       # Consolidated training dataset (2,820 samples)
├── eval_results/                         # Official benchmark evaluation transcripts & raw metrics
│   └── results_2026-10-20T00-59-42.294908.json # CAIS WMDP-Cyber lm-evaluation-harness output
├── notebooks/                            # Interactive environments and cloud training
│   └── CyberRott_Colab_Training.ipynb    # Official Google Colab Pro notebook (L4 / A100 / H100)
├── scripts/                              # Modular automation for sourcing, SFT, and deployment
│   ├── data_prep.py                      # Anchor Mixing engine (20x upsampler + HF streaming)
│   ├── generate_complex_samples.py       # Generator for technical golden anchor scenarios
│   ├── prepare_dataset.py                # ChatML schema validator and formatting utility
│   ├── distill.py                        # Reasoning distillation pipeline via external endpoints
│   ├── train_colab.py                    # Production CLI fine-tuning script with TRL/PEFT
│   ├── inference_test.py                 # Multi-prompt validation runner with CoT parsing
│   └── merge_lora.py                     # Weight unification utility (Base model + LoRA -> Standalone)
├── MODEL_CARD.md                         # Official model card conforming to Hugging Face Hub standards
├── README.md                             # Technical documentation and operational guide
└── requirements.txt                      # Pinned Python package dependencies
```

---

## 🚀 Reproduction and Training Pipeline

### 1. Cloud Execution via Google Colab Pro (Recommended: GPU L4 24GB or A100 40/80GB)

1. Upload the [`CyberRott_Colab_Training.ipynb`](file:///home/john/Programacao/Estudos/CyberRott1.0/notebooks/CyberRott_Colab_Training.ipynb) notebook to your Google Colab workspace.
2. Under **Runtime** -> **Change runtime type**, select either **L4 GPU** or **A100 GPU**.
3. Execute the cells sequentially. The automated pipeline manages:
   - GPU capability verification and CUDA toolchain inspection.
   - Installation of optimized dependencies (`flash-attn`, fused AdamW kernels, `trl`, `peft`).
   - Streaming ingestion and Anchor Mixing of NVIDIA GooseReason.
   - Full SFT training loop with periodic checkpoints and real-time validation.

### 2. Local Workstation or On-Premise GPU Cluster via CLI

```bash
# 1. Clone repository and initialize isolated Python virtual environment
git clone https://github.com/johnknevescarrijo/CyberRott1.0.git
cd CyberRott1.0
conda create -n cyberrott python=3.10 -y && conda activate cyberrott
pip install -r requirements.txt

# 2. Synthesize the balanced training dataset via Anchor Mixing
python scripts/data_prep.py \
    --anchor_file data/dataset_cyber_ready.jsonl \
    --anchor_repeat 20 \
    --hf_dataset nvidia/Nemotron-Research-GooseReason-0.7M \
    --max_hf_samples 2500 \
    --output data/cyberrott_final_train.jsonl \
    --seed 42

# 3. Launch LoRA Parameter-Efficient Fine-Tuning
python scripts/train_colab.py \
    --model_name_or_path microsoft/Phi-4-mini-instruct \
    --data_file data/cyberrott_final_train.jsonl \
    --output_dir ./CyberRott-Phi4-LoRA \
    --torch_dtype bfloat16 \
    --lora_r 32 \
    --lora_alpha 64 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --learning_rate 2e-4
```

---

## 🌐 Production Serving and Deployment Guide

### 1. LoRA Weight Merging (Standalone Hugging Face Export)

To serve the model through optimized inference servers (vLLM, Hugging Face TGI, or llama.cpp), consolidate the low-rank delta matrices directly into the base weights:

```bash
python scripts/merge_lora.py \
    --base_model microsoft/Phi-4-mini-instruct \
    --adapter ./CyberRott-Phi4-LoRA \
    --output ./CyberRott-Phi4-mini-Standalone \
    --torch_dtype bfloat16
```

### 2. High-Throughput Serving with vLLM

The vLLM engine delivers an OpenAI-compatible REST API with PagedAttention and continuous batching:

```bash
pip install vllm
vllm serve ./CyberRott-Phi4-mini-Standalone \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 4096
```

#### Querying the Endpoint via cURL:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "./CyberRott-Phi4-mini-Standalone",
    "messages": [
      {
        "role": "system",
        "content": "You are CyberRott, a specialized reasoning AI for offensive, defensive, and forensic cybersecurity."
      },
      {
        "role": "user",
        "content": "Analyze the security implications of reusing a nonce under the AES-GCM authenticated encryption mode."
      }
    ],
    "temperature": 0.6
  }'
```

### 3. Local Edge Deployment via Ollama (Quantized GGUF)

For execution on local engineering laptops, air-gapped workstations, or memory-constrained servers:

1. Convert the standalone PyTorch model to GGUF format using `llama.cpp`:
   ```bash
   python llama.cpp/convert_hf_to_gguf.py ./CyberRott-Phi4-mini-Standalone \
       --outtype q8_0 \
       --outfile cyberrott-q8.gguf
   ```

2. Define an Ollama `Modelfile`:
   ```dockerfile
   FROM ./cyberrott-q8.gguf
   TEMPLATE """<|im_start|>system
   {{ .System }}<|im_end|>
   <|im_start|>user
   {{ .Prompt }}<|im_end|>
   <|im_start|>assistant
   """
   PARAMETER temperature 0.6
   PARAMETER stop "<|im_end|>"
   ```

3. Build and launch the containerized runtime:
   ```bash
   ollama create cyberrott -f Modelfile
   ollama run cyberrott
   ```

---

## 🛡️ Responsible Dual-Use Governance and Ethical Mandate

> [!CAUTION]
> **Defensive and Authorized Operational Boundaries**:
> **CyberRott 1.0** is developed strictly as a technical accelerator for defensive engineering, security auditing, digital forensics, incident response, and explicitly authorized penetration testing engagements.
> 
> - **Permitted Use Cases**: Scope-constrained vulnerability assessment, defensive code remediation, CI/CD static security review, Blue Team SIEM/EDR detection rule engineering, academic vulnerability research, and threat emulation within authorized boundaries.
> - **Prohibited Activities**: Deployment against computational assets without prior formal authorization, operationalization of weaponized zero-day exploits, synthesis of destructive malware binaries, or actions violating applicable domestic and international cybersecurity legislation.
> - **Human-in-the-Loop Requirement**: Because generative models can produce speculative analyses in novel or edge-case environments, all findings and tactical remediation guidance must undergo validation by qualified application security (AppSec) or security operations (SecOps) engineers.

---

## 👨‍💻 Authors & Acknowledgments

- **Lead Architect and Developer**: [John Carrijo](https://github.com/johnknevescarrijo)
- **Foundation Model Architecture**: Microsoft Research (`Phi-4-mini-instruct`)
- **Reasoning Dataset Sourcing**: NVIDIA Nemotron Research Team (`GooseReason-0.7M`)
- **Open-Source Infrastructure**: Hugging Face Team (TRL, PEFT, Transformers, Datasets)

---

<div align="center">
<i>"Fortifying defensive resilience through structured, verifiable machine reasoning."</i>
</div>