# Artigos Fundamentais (Papers) do CyberRott

Esta pasta reúne os artigos científicos seminais que fundamentam as técnicas implementadas no pipeline do **CyberRott**:

---

### 1. LoRA (Low-Rank Adaptation)
* **Arquivo:** [`01_LoRA_Hu_et_al_2021.pdf`](file:///home/john/Programacao/Estudos/CyberRott1.0/Papers/01_LoRA_Hu_et_al_2021.pdf)
* **Título:** *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021)
* **arXiv:** [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
* **O que estuda:** Como congelar os pesos $W_0$ da rede e treinar apenas matrizes decompostas de baixo posto ($B \cdot A$), reduzindo drasticamente o consumo de VRAM e permitindo fusão direta dos pesos após o treino.
* **Onde se conecta no seu código:**
  - Fusão de pesos: [`merge_lora.py`](file:///home/john/Programacao/Estudos/CyberRott1.0/scripts/merge_lora.py)
  - Configurações de `r`, `lora_alpha` e `target_modules` em [`train_colab.py`](file:///home/john/Programacao/Estudos/CyberRott1.0/scripts/train_colab.py) e [`train_grpo.py`](file:///home/john/Programacao/Estudos/CyberRott1.0/scripts/train_grpo.py).

---

### 2. STaR (Self-Taught Reasoner)
* **Arquivo:** [`02_STaR_Zelikman_et_al_2022.pdf`](file:///home/john/Programacao/Estudos/CyberRott1.0/Papers/02_STaR_Zelikman_et_al_2022.pdf)
* **Título:** *STaR: Bootstrapping Reasoning With Reasoning* (Zelikman et al., 2022 - Stanford)
* **arXiv:** [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
* **O que estuda:** Como fazer o modelo gerar justificativas (racionais) passo a passo, filtrar apenas as tentativas que levaram à resposta final correta (usando um verificador/ground truth) e retreinar o modelo iterativamente nesse subconjunto.
* **Onde se conecta no seu código:**
  - Loop de treinamento do STaR: [`train_star.py`](file:///home/john/Programacao/Estudos/CyberRott1.0/scripts/train_star.py)
  - Motor de validação das soluções: [`star_verifier.py`](file:///home/john/Programacao/Estudos/CyberRott1.0/scripts/star_verifier.py)

---

### 3. DeepSeekMath & GRPO (Group Relative Policy Optimization)
* **Arquivo:** [`03_DeepSeekMath_GRPO_2024.pdf`](file:///home/john/Programacao/Estudos/CyberRott1.0/Papers/03_DeepSeekMath_GRPO_2024.pdf)
* **Título:** *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (Shao et al., 2024)
* **arXiv:** [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
* **O que estuda:** A introdução formal do algoritmo **GRPO**. Mostra como o GRPO elimina a rede crítica (Critic / Value Model) necessária no PPO clássico, amostrando grupos de respostas para o mesmo prompt e calculando vantagens relativas à média e desvio padrão do grupo.
* **Onde se conecta no seu código:**
  - Pipeline de RL com TRL GRPO: [`train_grpo.py`](file:///home/john/Programacao/Estudos/CyberRott1.0/scripts/train_grpo.py)

---

### 4. DeepSeek-R1 (Incentivando Raciocínio Puro via RL)
* **Arquivo:** [`04_DeepSeek_R1_Reasoning_GRPO_2025.pdf`](file:///home/john/Programacao/Estudos/CyberRott1.0/Papers/04_DeepSeek_R1_Reasoning_GRPO_2025.pdf)
* **Título:** *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (DeepSeek-AI, 2025)
* **arXiv:** [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
* **O que estuda:** Aplicação em larga escala do GRPO diretamente em modelos base (R1-Zero) e com SFT frio (R1), demonstrando a emergência de pensamento reflexivo ("aha moments") e cadeia de raciocínio detalhada sem anotação humana massiva.
* **Onde se conecta no seu código:**
  - Estrutura de tags `<think> ... </think>` e funções de recompensa de formato e precisão em [`train_grpo.py`](file:///home/john/Programacao/Estudos/CyberRott1.0/scripts/train_grpo.py).

---

### 5. Phi-4-Mini Technical Report
* **Arquivo:** [`05_Phi4_Mini_Technical_Report_2025.pdf`](file:///home/john/Programacao/Estudos/CyberRott1.0/Papers/05_Phi4_Mini_Technical_Report_2025.pdf)
* **Título:** *Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs* (Microsoft, 2025)
* **arXiv:** [arXiv:2503.01743](https://arxiv.org/abs/2503.01743)
* **O que estuda:** A arquitetura do modelo base de 3.8B que você está utilizando: 32 camadas, Grouped Query Attention (GQA), vocabulário expandido (200K), Mixture-of-LoRAs e seleção de dados sintéticos de alta densidade.
* **Onde se conecta no seu código:**
  - Conversão e alinhamento de tensores GGUF: [`convert_hf_to_gguf.py`](file:///home/john/Programacao/Estudos/CyberRott1.0/scripts/convert_hf_to_gguf.py)
  - Configuração do modelo base e inferência: [`inference_test.py`](file:///home/john/Programacao/Estudos/CyberRott1.0/scripts/inference_test.py) e [`Modelfile`](file:///home/john/Programacao/Estudos/CyberRott1.0/Modelfile)

---

### 6. Phi-4 Technical Report (Base 14B)
* **Arquivo:** [`06_Phi4_Technical_Report_2024.pdf`](file:///home/john/Programacao/Estudos/CyberRott1.0/Papers/06_Phi4_Technical_Report_2024.pdf)
* **Título:** *Phi-4 Technical Report* (Microsoft, 2024)
* **arXiv:** [arXiv:2412.08905](https://arxiv.org/abs/2412.08905)
* **O que estuda:** Detalhes metodológicos da Microsoft sobre pré-treino, pós-treino (DPO, SFT) e filtragem sintética de dados para modelos de alta capacidade de raciocínio.
