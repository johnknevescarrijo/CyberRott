"""
Script de Destilação Sintética para o CyberRott 1.0
Gera dados de treino com raciocínio passo a passo (<think>...</think>)
usando modelos mestres (DeepSeek V3 / R1 ou GLM via API compatível com OpenAI).
"""

import os
import json
import argparse
from typing import List, Dict
from openai import OpenAI
from tqdm import tqdm

SYSTEM_PROMPT = """Você é um especialista sênior em segurança da informação, análise de vulnerabilidades, engenharia reversa e operações Red/Blue Team.
Ao responder à pergunta ou analisar o cenário de cibersegurança fornecido:
1. Pense detalhadamente passo a passo dentro das tags <think> e </think>. Analise vetores de ataque, hipóteses, montagem/código, RFCs e mitigações.
2. Após fechar </think>, forneça a solução técnica precisa, prática e direta ao ponto."""

SAMPLE_PROMPTS = [
    "Analise o seguinte trecho em C vulnerável a Buffer Overflow e explique passo a passo como o ponteiro de retorno é sobrescrito no stack frame x86_64:\n\nvoid vuln() {\n    char buffer[64];\n    gets(buffer);\n}",
    "Explique como identificar um ataque de Kerberoasting através da análise de logs de eventos do Windows (Event ID 4769) e quais filtros usar no Splunk/Elastic.",
    "Dado o binário ELF descompilado com UPX modificado, quais as etapas para desempacotar manualmente na memória via GDB?",
    "Como funciona a exploração de deserialização insegura em Java utilizando a cadeia de gadgets do CommonsCollections1?",
    "Quais as diferenças arquiteturais entre EDR bypass por Direct System Calls (SysWhispers) e Unhooking de ntdll.dll?"
]

def distill_examples(
    prompts: List[str],
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    model_name: str = "deepseek-reasoner",
    output_file: str = "dataset_cyber_distilled.jsonl"
):
    client = OpenAI(api_key=api_key, base_url=base_url)
    dataset = []

    print(f"Destilando {len(prompts)} exemplos usando o modelo {model_name}...")

    with open(output_file, "a", encoding="utf-8") as f_out:
        for prompt in tqdm(prompts):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.6,
                )

                # Se o modelo mestre tiver campo de raciocínio nativo (como deepseek-reasoner/R1)
                message = response.choices[0].message
                reasoning = getattr(message, "reasoning_content", None)
                content = message.content

                if reasoning:
                    full_response = f"<think>\n{reasoning}\n</think>\n\n{content}"
                else:
                    full_response = content

                data_entry = {
                    "messages": [
                        {"role": "system", "content": "Você é o CyberRott, uma IA especialista em segurança defensiva, ofensiva e análise de ameaças com raciocínio profundo."},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": full_response}
                    ]
                }

                f_out.write(json.dumps(data_entry, ensure_ascii=False) + "\n")
                f_out.flush()

            except Exception as e:
                print(f"Erro ao processar prompt: {e}")

    print(f"Destilação concluída! Amostras gravadas em {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Destilador de Raciocínio CyberRott")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY", ""), help="Chave de API")
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com", help="Base URL da API")
    parser.add_argument("--model", type=str, default="deepseek-reasoner", help="Nome do modelo professor")
    parser.add_argument("--output", type=str, default="dataset_cyber_distilled.jsonl", help="Arquivo JSONL de saída")
    args = parser.parse_args()

    if not args.api_key:
        print("[Aviso] Nenhuma API KEY informada. Gravando arquivo de exemplo com dados sintéticos locais para teste imediato...")
        dummy_entry = {
            "messages": [
                {"role": "system", "content": "Você é o CyberRott, especialista em cibersegurança."},
                {"role": "user", "content": SAMPLE_PROMPTS[0]},
                {"role": "assistant", "content": "<think>\n1. A função vuln() aloca 64 bytes para o array 'buffer'.\n2. 'gets()' não verifica limites de buffer (bounds checking).\n3. O stack frame armazena: buffer (64 bytes) + padding + Saved RBP (8 bytes) + Return Address (8 bytes).\n4. Entrada > 72 bytes sobrescreve o endereço de retorno.\n</think>\n\nA vulnerabilidade decorre do uso de `gets()`. Ao enviar mais de 72 bytes, o fluxo de execução é sequestrado no `ret`."}
            ]
        }
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json.dumps(dummy_entry, ensure_ascii=False) + "\n")
        print(f"Exemplo de teste gerado em {args.output}")
    else:
        distill_examples(
            prompts=SAMPLE_PROMPTS,
            api_key=args.api_key,
            base_url=args.base_url,
            model_name=args.model,
            output_file=args.output
        )
