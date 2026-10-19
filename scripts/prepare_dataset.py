"""
Script utilitário para validar, corrigir e converter os dados destilados do GLM/DeepSeek
para o formato oficial de conversação (ChatML / Phi-4 Template).
"""

import json
import os
import glob
from typing import List, Dict

def convert_and_validate(
    input_path: str = "/home/john/Programacao/Estudos/Machine_Learning/dataset_cyber/dataset_seguranca.jsonl",
    output_path: str = "./dataset_cyber_ready.jsonl"
):
    if not os.path.exists(input_path):
        print(f"[Erro] Arquivo de entrada não encontrado: {input_path}")
        return

    valid_samples = 0
    total_tokens_approx = 0

    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except Exception as e:
                print(f"[Linha {idx}] Erro de parsing JSON: {e}")
                continue

            # Caso 1: Formato antigo (instruction, input, reasoning_trace, final_answer)
            if "reasoning_trace" in data and "final_answer" in data:
                instruction = data.get("instruction", "").strip()
                user_input = data.get("input", "").strip()
                reasoning = data.get("reasoning_trace", "").strip()
                final_answer = data.get("final_answer", "").strip()

                # Combinar user prompt
                if user_input:
                    prompt = f"{instruction}\n\n```\n{user_input}\n```" if instruction else user_input
                else:
                    prompt = instruction

                # Garantir que a tag <think> e </think> estejam fechadas
                if not reasoning.startswith("<think>"):
                    reasoning = f"<think>\n{reasoning}"
                if not reasoning.endswith("</think>"):
                    reasoning = f"{reasoning}\n</think>"

                assistant_content = f"{reasoning}\n\n{final_answer}"

                formatted = {
                    "messages": [
                        {"role": "system", "content": "Você é o CyberRott, uma IA especialista em segurança ofensiva, defensiva e análise de vulnerabilidades com raciocínio analítico profundo."},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": assistant_content}
                    ]
                }
            # Caso 2: Já no formato de messages
            elif "messages" in data:
                formatted = data
            else:
                print(f"[Linha {idx}] Formato desconhecido, ignorando.")
                continue

            f_out.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            valid_samples += 1
            # Estimativa de tokens grosseira (~4 caracteres por token)
            text_len = sum(len(m["content"]) for m in formatted["messages"])
            total_tokens_approx += text_len // 4

    print("=" * 60)
    print(f"Resultado da Conversão e Validação:")
    print(f"- Amostras processadas: {valid_samples}")
    print(f"- Média de tokens estimados por amostra: {total_tokens_approx // max(valid_samples, 1)}")
    print(f"- Arquivo pronto gerado em: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    convert_and_validate()
