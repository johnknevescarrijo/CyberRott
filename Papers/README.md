# Artigos Fundamentais (Papers) do CyberRott

Esta pasta reúne os artigos científicos seminais que fundamentam as técnicas implementadas no pipeline do **CyberRott**:

---

### 1. LoRA (Low-Rank Adaptation)
* **Arquivo:** [`01_LoRA_Hu_et_al_2021.pdf`](file:///home/john/Programacao/Estudos/CyberRott/Papers/01_LoRA_Hu_et_al_2021.pdf)
* **Título:** *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021)
* **arXiv:** [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
* **O que estuda:** Como congelar os pesos $W_0$ da rede e treinar apenas matrizes decompostas de baixo posto ($B \cdot A$), reduzindo drasticamente o consumo de VRAM e permitindo fusão direta dos pesos após o treino.
* **Onde se conecta no seu código:**
  - Fusão de pesos: [`merge_lora.py`](file:///home/john/Programacao/Estudos/CyberRott/scripts/merge_lora.py)
  - Configurações de `r`, `lora_alpha` e `target_modules` em [`train_colab.py`](file:///home/john/Programacao/Estudos/CyberRott/scripts/train_colab.py) e [`train_grpo.py`](file:///home/john/Programacao/Estudos/CyberRott/scripts/train_grpo.py).

---

### 2. STaR (Self-Taught Reasoner)
* **Arquivo:** [`02_STaR_Zelikman_et_al_2022.pdf`](file:///home/john/Programacao/Estudos/CyberRott/Papers/02_STaR_Zelikman_et_al_2022.pdf)
* **Título:** *STaR: Bootstrapping Reasoning With Reasoning* (Zelikman et al., 2022 - Stanford)
* **arXiv:** [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
* **O que estuda:** Como fazer o modelo gerar justificativas (racionais) passo a passo, filtrar apenas as tentativas que levaram à resposta final correta (usando um verificador/ground truth) e retreinar o modelo iterativamente nesse subconjunto.
* **Onde se conecta no seu código:**
  - Loop de treinamento do STaR: [`train_star.py`](file:///home/john/Programacao/Estudos/CyberRott/scripts/train_star.py)
  - Motor de validação das soluções: [`star_verifier.py`](file:///home/john/Programacao/Estudos/CyberRott/scripts/star_verifier.py)

---

### 3. DeepSeekMath & GRPO (Group Relative Policy Optimization)
* **Arquivo:** [`03_DeepSeekMath_GRPO_2024.pdf`](file:///home/john/Programacao/Estudos/CyberRott/Papers/03_DeepSeekMath_GRPO_2024.pdf)
* **Título:** *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (Shao et al., 2024)
* **arXiv:** [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
* **O que estuda:** A introdução formal do algoritmo **GRPO**. Mostra como o GRPO elimina a rede crítica (Critic / Value Model) necessária no PPO clássico, amostrando grupos de respostas para o mesmo prompt e calculando vantagens relativas à média e desvio padrão do grupo.
* **Onde se conecta no seu código:**
  - Pipeline de RL com TRL GRPO: [`train_grpo.py`](file:///home/john/Programacao/Estudos/CyberRott/scripts/train_grpo.py)

---

### 4. DeepSeek-R1 (Incentivando Raciocínio Puro via RL)
* **Arquivo:** [`04_DeepSeek_R1_Reasoning_GRPO_2025.pdf`](file:///home/john/Programacao/Estudos/CyberRott/Papers/04_DeepSeek_R1_Reasoning_GRPO_2025.pdf)
* **Título:** *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (DeepSeek-AI, 2025)
* **arXiv:** [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
* **O que estuda:** Aplicação em larga escala do GRPO diretamente em modelos base (R1-Zero) e com SFT frio (R1), demonstrando a emergência de pensamento reflexivo ("aha moments") e cadeia de raciocínio detalhada sem anotação humana massiva.
* **Onde se conecta no seu código:**
  - Estrutura de tags `<think> ... </think>` e funções de recompensa de formato e precisão em [`train_grpo.py`](file:///home/john/Programacao/Estudos/CyberRott/scripts/train_grpo.py).

---

### 5. Phi-4-Mini Technical Report
* **Arquivo:** [`05_Phi4_Mini_Technical_Report_2025.pdf`](file:///home/john/Programacao/Estudos/CyberRott/Papers/05_Phi4_Mini_Technical_Report_2025.pdf)
* **Título:** *Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs* (Microsoft, 2025)
* **arXiv:** [arXiv:2503.01743](https://arxiv.org/abs/2503.01743)
* **O que estuda:** A arquitetura do modelo base de 3.8B que você está utilizando: 32 camadas, Grouped Query Attention (GQA), vocabulário expandido (200K), Mixture-of-LoRAs e seleção de dados sintéticos de alta densidade.
* **Onde se conecta no seu código:**
  - Conversão e alinhamento de tensores GGUF: [`convert_hf_to_gguf.py`](file:///home/john/Programacao/Estudos/CyberRott/scripts/convert_hf_to_gguf.py)
  - Configuração do modelo base e inferência: [`inference_test.py`](file:///home/john/Programacao/Estudos/CyberRott/scripts/inference_test.py) e [`Modelfile`](file:///home/john/Programacao/Estudos/CyberRott/Modelfile)

---

### 6. Phi-4 Technical Report (Base 14B)
* **Arquivo:** [`06_Phi4_Technical_Report_2024.pdf`](file:///home/john/Programacao/Estudos/CyberRott/Papers/06_Phi4_Technical_Report_2024.pdf)
* **Título:** *Phi-4 Technical Report* (Microsoft, 2024)
* **arXiv:** [arXiv:2412.08905](https://arxiv.org/abs/2412.08905)
* **O que estuda:** Detalhes metodológicos da Microsoft sobre pré-treino, pós-treino (DPO, SFT) e filtragem sintética de dados para modelos de alta capacidade de raciocínio.

---

### 7. DoRA: Weight-Decomposed Low-Rank Adaptation
* **Título:** *DoRA: Weight-Decomposed Low-Rank Adaptation* (Liu et al., 2024)
* **arXiv:** [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)
* **O que estuda:** Decomposição dos pesos em vetor de magnitude $m$ e matriz de direção $V$, aplicando atualização de baixo posto exclusivamente na direção ($\Delta W = m \frac{W_0 + BA}{\|W_0 + BA\|_F}$). Isso estabiliza o aprendizado, reduz a lacuna entre LoRA e fine-tuning completo e melhora a expressividade matemática e de código.
* **Onde se conecta no seu código:**
  - Suporte nativo a DoRA via `--use_dora`, `--dora_rank 64`, `--dora_alpha 128` em [`train_grpo.py`](file:///home/john/Programacao/Estudos/CyberRott/scripts/train_grpo.py), [`train_star.py`](file:///home/john/Programacao/Estudos/CyberRott/scripts/train_star.py) e [`train_colab.py`](file:///home/john/Programacao/Estudos/CyberRott/scripts/train_colab.py).

---

### 8. Let's Verify Step by Step (Process-Supervised Reward Models - PRM)
* **Título:** *Let's Verify Step by Step* (Lightman et al., 2023 - OpenAI)
* **arXiv:** [arXiv:2305.20050](https://arxiv.org/abs/2305.20050)
* **O que estuda:** Comparação entre supervisão de resultado final (Outcome-supervised Reward Models - ORM) e supervisão por passos intermediários (Process-supervised Reward Models - PRM). Mostra que pontuar cada passo dedutivo individualmente reduz drasticamente alucinações, mitiga reward hacking e permite aos modelos detectar e corrigir erros em problemas dedutivos complexos.
* **Onde se conecta no seu código:**
  - Verificador de passos analíticos: `verify_prm_steps` e `compute_prm_step_score` em [`star_verifier.py`](file:///home/john/Programacao/Estudos/CyberRott/scripts/star_verifier.py).
  - Quarta função de recompensa em RL: `prm_step_reward` em [`train_grpo.py`](file:///home/john/Programacao/Estudos/CyberRott/scripts/train_grpo.py).
  - Filtragem e pontuação de trajetórias sintéticas: [`train_star.py`](file:///home/john/Programacao/Estudos/CyberRott/scripts/train_star.py).

---

### 9. Self-Consistency & Test-Time Compute Scaling
* **Título:** *Self-Consistency Improves Chain of Thought Reasoning in Language Models* (Wang et al., 2022 - Google Research) / *Scaling LLM Test-Time Compute Optimally* (Snell et al., 2024 - UC Berkeley)
* **arXiv:** [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) & [arXiv:2408.03314](https://arxiv.org/abs/2408.03314)
* **O que estuda:** Como alocar computação adicional no momento da inferência (Test-Time Compute) em vez de apenas no pré-treinamento. Amostra múltiplos caminhos de raciocínio diversos em paralelo e agrega as respostas finais por maioria simples (Majority Voting), maioria ponderada por verificador (PRM-Weighted Self-Consistency) ou seleção do melhor candidato (Best-of-$N$).
* **Onde se conecta no seu código:**
  - Seleção Best-of-$N$: `select_best_of_n` em [`eval_wmdp_vllm.py`](file:///home/john/Programacao/Estudos/CyberRott/scripts/eval_wmdp_vllm.py).
  - Agregação por maioria e self-consistency: `aggregate_majority_vote` e `aggregate_test_time_compute` em [`eval_wmdp_vllm.py`](file:///home/john/Programacao/Estudos/CyberRott/scripts/eval_wmdp_vllm.py).
  - Amostragem paralela acelerada via vLLM (`--num_samples`, `--ttc_strategy`).
