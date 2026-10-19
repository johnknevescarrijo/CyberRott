#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerador de Amostras Complexas de Cibersegurança para o CyberRott 1.0.
Constrói 16 exemplos de alta densidade analítica cobrindo:
1. SQL Injection (Python/SQLite)
2. Conversão de Tipos / Integer Overflow -> Buffer Overflow (C)
3. Use-After-Free em Gestão de Sessão (C++)
4. Deserialização Insegura com ObjectInputStream (Java)
5. SSRF com Bypass de Filtros e IMDSv1/v2 (Python)
6. Race Condition TOCTOU em Transação Financeira (Go)
7. Kerberos Constrained Delegation / S4U (Active Directory)
8. User-Mode Hooking vs Kernel Callbacks / ETW-Ti (Windows Internals)
9. Process Hollowing / Sysmon & Regra Sigma (Forense)
10. Format String Vulnerability em Binário (C / RevEng)
11. Reentrancy em Smart Contract e CEI Pattern (Solidity)
12. DNS Tunneling e Exfiltração de Dados (Forense de Rede / KQL)
13. Confusão de Algoritmo em JWT RS256 -> HS256 (Node.js)
14. Escape de Container Docker via socket / CAP_SYS_ADMIN (Linux Containers)
15. Reuso de Nonce em AES-GCM e GHASH Forgery (Criptografia)
16. Path Traversal e Local File Inclusion (PHP)
"""

import json
import os

OUTPUT_JSONL = "/home/john/Programacao/Estudos/Machine_Learning/dataset_cyber/dataset_seguranca.jsonl"

def get_all_samples():
    samples = []

    # -------------------------------------------------------------------------
    # EXEMPLO 1: SQL Injection em Python (Amostra Original Mantida e Formatada)
    # -------------------------------------------------------------------------
    samples.append({
        "instruction": "Analise o código abaixo em busca de vulnerabilidades de segurança. Apresente seu raciocínio passo a passo, identificando pontos de entrada, fluxo de dados, sinks perigosos e avaliando a sanitização existente. Conclua com o tipo de vulnerabilidade (se houver), impacto potencial e recomendação de correção.",
        "input": "import sqlite3\ndef get_user(username):\n    conn = sqlite3.connect('users.db')\n    cursor = conn.cursor()\n    query = f\"SELECT * FROM users WHERE username = '{username}'\"\n    cursor.execute(query)\n    return cursor.fetchone()",
        "reasoning_trace": "<think>\n"
            "**Análise de Superfície**: O código é uma função Python chamada `get_user` que recebe um parâmetro `username` e retorna o registro correspondente da tabela `users` em um banco SQLite local (`users.db`). O propósito legítimo é a busca de um usuário pelo seu nome.\n\n"
            "**Identificação de Pontos de Entrada / Vetores**: O ponto de entrada é o parâmetro `username` da função (linha 2). Como não há nenhum código de chamada visível, deve-se assumir o pior caso: `username` é total ou parcialmente controlado por dados externos (input de usuário via API HTTP, formulário, parâmetro de requisição, etc.). Não há qualquer restrição de origem documentada no código.\n\n"
            "**Mapeamento de Fluxo / Mecânica Interna**: O dado `username` flui diretamente, sem qualquer transformação intermediária, para a linha 5, onde é interpolado na string `query` via f-string: `f\"SELECT * FROM users WHERE username = '{username}'\"`. A string construída é passada imediatamente como único argumento para `cursor.execute(query)` na linha 6. O fluxo é entrada → interpolação em string SQL → execução, sem barreiras no meio do caminho.\n\n"
            "**Detecção de Sinks Perigosos / Falhas de Controle**: O sink é `cursor.execute(query)` (linha 6). Este é o sink clássico de SQL Injection: a query SQL é montada por concatenação/interpolação de string em vez de usar queries parametrizadas. A interpolação por f-string em Python não tem qualquer semântica de escape SQL — o conteúdo de `username` é inserido literalmente na query.\n\n"
            "**Avaliação de Sanitização e Mitigações**: Não existe nenhuma validação, sanitização, escape ou uso de placeholders entre o ponto de entrada e o sink. Não há: (a) lista de permissões (allowlist) de caracteres; (b) verificação de tipo/formato do `username`; (c) escape de aspas simples; (d) uso de queries parametrizadas (`execute` com `?`). A avaliação é: sanitização inexistente.\n\n"
            "**Conclusão da Vulnerabilidade / Anomalia**: O código é VULNERÁVEL. A interpolação direta de `username` na string SQL permite que um atacante feche a aspa simples, injete SQL arbitrário e comente o restante da query. Por exemplo, o payload `admin'--` produz: `SELECT * FROM users WHERE username = 'admin'--'`, retornando o registro do admin sem senha. Payloads como `' OR '1'='1` retornam todos os usuários. Trata-se de SQL Injection (CWE-89). Vale notar: o `sqlite3` do Python suporta apenas uma statements por `execute()`, o que reduz (mas não elimina) o risco de stacked queries; extração de dados via UNION, boolean-based e time-based blind injection permanecem totalmente viáveis.\n\n"
            "**Impacto Potencial**: Pior cenário: (1) bypass de autenticação — um atacante pode recuperar o registro de qualquer usuário, incluindo hashes de senhas; (2) extração completa do banco — com UNION-based ou blind SQL Injection, é possível enumerar todo o schema e exfiltrar todas as tabelas; (3) se a aplicação usar este padrão em outras funções com permissões maiores, o impacto escala para modificação/deleção de dados (via UPDATE/DELETE injetados em outros fluxos). No SQLite não há `SLEEP()`, então blind injection seria baseada em resposta booleana ou em erros, não em timing.\n"
            "</think>",
        "final_answer": "FATO: A função `get_user` é vulnerável a SQL Injection (CWE-89). O parâmetro `username` (linha 2) é interpolado diretamente na query SQL via f-string (linha 5) e executado sem qualquer sanitização em `cursor.execute(query)` (linha 6). Um atacante que controle `username` pode injetar SQL arbitrário, bypassar autenticação (ex: payload `admin'--`) e exfiltrar dados do banco via UNION-based/blind injection.\n\n"
            "DETECÇÃO / TELEMETRIA: Monitorar logs de consultas SQL anômalas contendo meta-caracteres como `'--`, `' OR '1'='1` ou operadores `UNION SELECT`. Em nível de rede ou WAF, configurar regras do ModSecurity Core Rule Set (CRS 942100) para inspeção de parâmetros HTTP.\n\n"
            "SUGESTÃO DE CORREÇÃO: Substituir a interpolação por query parametrizada, que é a correção objetiva e definitiva:\n"
            "```python\n"
            "cursor.execute(\"SELECT * FROM users WHERE username = ?\", (username,))\n"
            "```\n"
            "Adicionalmente (defesa em profundidade): validar `username` com uma allowlist de caracteres (ex: alfanuméricos, `^[a-zA-Z0-9_.-]{3,32}$`) e limitar privilégios da conta de conexão ao banco."
    })

    # -------------------------------------------------------------------------
    # EXEMPLO 2: C / Memory Safety (Signed/Unsigned Conversion CWE-195/681 -> BOF)
    # -------------------------------------------------------------------------
    samples.append({
        "instruction": "Realize uma auditoria de segurança aprofundada no código C de tratamento de pacotes de rede abaixo. Efetue uma análise de fluxo e conversão de tipos (taint analysis), identificando se há discrepâncias de sinal ou estouro de inteiros que possam comprometer a integridade da memória. Apresente a causa raiz técnica, impacto e forneça a correção segura completa.",
        "input": "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <unistd.h>\n#include <sys/socket.h>\n\n#define MAX_PAYLOAD 1024\n\nvoid process_packet(int sock) {\n    short payload_len;\n    char buffer[MAX_PAYLOAD];\n\n    // Lê o cabeçalho de comprimento enviado pelo cliente (2 bytes)\n    if (recv(sock, &payload_len, sizeof(payload_len), 0) != sizeof(payload_len)) {\n        return;\n    }\n\n    // Validação ingênua de limite\n    if (payload_len > MAX_PAYLOAD) {\n        printf(\"[!] Tamanho excede o limite permitido.\\n\");\n        return;\n    }\n\n    // Leitura do corpo da mensagem para o buffer local na stack\n    ssize_t bytes_read = recv(sock, buffer, payload_len, 0);\n    if (bytes_read > 0) {\n        buffer[bytes_read - 1] = '\\0';\n        printf(\"[*] Pacote recebido: %s\\n\", buffer);\n    }\n}",
        "reasoning_trace": "<think>\n"
            "**Análise de Superfície**: A função C `process_packet` recebe um descritor de socket `sock`, lê um cabeçalho de 2 bytes indicando o tamanho do pacote para a variável local `short payload_len`, realiza uma verificação de limite superior contra `MAX_PAYLOAD` (1024) e, em seguida, executa uma segunda leitura de socket gravando os dados no array `char buffer[MAX_PAYLOAD]` alocado na stack.\n\n"
            "**Identificação de Pontos de Entrada / Vetores**: O ponto de entrada é a conexão de rede representada por `sock`. O dado lido em `&payload_len` é integralmente fornecido por uma entidade remota externa não autenticada e deve ser tratado como completamente inseguro (tainted).\n\n"
            "**Mapeamento de Fluxo / Mecânica Interna**: A variável `payload_len` é declarada como `short`, que em arquiteturas padrão é um inteiro com sinal de 16 bits (`signed short`, variando de -32768 a 32767). Quando o invasor transmite o valor binário `0xFFFF`, a variável assume o valor decimal `-1`. A condicional de validação `if (payload_len > MAX_PAYLOAD)` compara `-1 > 1024`, o que é falso, permitindo que a execução prossiga. Logo em seguida, `payload_len` é passado como o terceiro parâmetro da chamada `recv(sock, buffer, payload_len, 0)`. Conforme o padrão POSIX, a assinatura de `recv` define o terceiro argumento como `size_t len`, que é um tipo inteiro não assinado (unsigned) de 64 bits em sistemas x86_64. Ocorre uma conversão de tipo implícita com extensão de sinal (Signed to Unsigned Conversion / Sign Extension - CWE-195 e CWE-681): o valor `-1` é promovido e convertido para `0xFFFFFFFFFFFFFFFF` (`SIZE_MAX`), representando aproximadamente 18.4 exabytes.\n\n"
            "**Detecção de Sinks Perigosos / Falhas de Controle**: O sink perigoso é a função `recv(sock, buffer, payload_len, 0)`. Como o buffer alocado na stack possui exatamente 1024 bytes e a chamada `recv` aceita transferir até `SIZE_MAX` bytes, o sistema operacional gravará dados recebidos da rede sequencialmente na stack, ultrapassando os limites do array local.\n\n"
            "**Avaliação de Sanitização e Mitigações**: A sanitização é falha e incompleta: verifica apenas o limite superior (`payload_len > MAX_PAYLOAD`), omitindo completamente a verificação de valores negativos (`payload_len <= 0`). Além disso, a escolha do tipo `short` com sinal para controle de tamanho de buffer é uma violação de tipagem defensiva.\n\n"
            "**Conclusão da Vulnerabilidade / Anomalia**: Trata-se de uma vulnerabilidade de conversão inadequada de tipos inteiros com sinal para sem sinal (CWE-195 / CWE-681), que resulta diretamente em um Stack-Based Buffer Overflow clássico (CWE-121).\n\n"
            "**Impacto Potencial**: Sobrescrita da memória da pilha (Saved Frame Pointer e Return Address - RIP/EIP). Um invasor pode construir uma cadeia de ROP (Return-Oriented Programming) para desviar o fluxo de execução e obter Execução Remota de Código (RCE) no contexto de execução do processo, ou no mínimo causar Denial of Service (DoS) por interrupção catastrófica de processo (SIGSEGV / abort).\n"
            "</think>",
        "final_answer": "FATO: O código é vulnerável a conversão incorreta de sinal de inteiro (Signed-to-Unsigned Conversion / CWE-195 e CWE-681), levando a Stack-Based Buffer Overflow (CWE-121). A variável `payload_len` é declarada como `short` (com sinal) e a verificação `if (payload_len > MAX_PAYLOAD)` não testa valores negativos. Se um valor negativo como `-1` (`0xFFFF`) for fornecido, a verificação passa e o valor é promovido para `size_t` na chamada `recv()`, convertendo-se em `SIZE_MAX`, o que permite sobrescrever a pilha além dos 1024 bytes alocados.\n\n"
            "DETECÇÃO / TELEMETRIA: Instrumentação com AddressSanitizer (`gcc -fsanitize=address,undefined`) gerando relatório imediato de `stack-buffer-overflow`. Em produção, monitorar crashes via `coredumpctl` ou `/var/log/messages` com alertas do compilador `*** stack smashing detected ***: terminated` disparados pela proteção `__stack_chk_fail`.\n\n"
            "SUGESTÃO DE CORREÇÃO: Utilizar tipos não assinados de largura fixa (`uint16_t` ou `size_t`), converter da ordem de rede com `ntohs()`, verificar explicitamente os limites de intervalo (rejeitando tamanhos nulos ou que excedam `sizeof(buffer)`), e limitar a leitura ao tamanho estrito do array de destino:\n"
            "```c\n"
            "#include <stdint.h>\n"
            "#include <arpa/inet.h>\n\n"
            "void process_packet_fixed(int sock) {\n"
            "    uint16_t net_len;\n"
            "    char buffer[MAX_PAYLOAD];\n\n"
            "    if (recv(sock, &net_len, sizeof(net_len), MSG_WAITALL) != sizeof(net_len)) {\n"
            "        return;\n"
            "    }\n\n"
            "    uint16_t payload_len = ntohs(net_len);\n"
            "    if (payload_len == 0 || payload_len >= sizeof(buffer)) {\n"
            "        printf(\"[!] Tamanho de pacote invalido: %u\\n\", payload_len);\n"
            "        return;\n"
            "    }\n\n"
            "    ssize_t bytes_read = recv(sock, buffer, payload_len, MSG_WAITALL);\n"
            "    if (bytes_read > 0) {\n"
            "        buffer[bytes_read] = '\\0';\n"
            "        printf(\"[*] Pacote recebido: %s\\n\", buffer);\n"
            "    }\n"
            "}\n"
            "```"
    })

    # -------------------------------------------------------------------------
    # EXEMPLO 3: C++ / Memory Safety (Use-After-Free CWE-416)
    # -------------------------------------------------------------------------
    samples.append({
        "instruction": "Analise o trecho de código C++ a seguir responsável pelo ciclo de vida de conexões e sessões de usuários. Identifique falhas de gerenciamento de memória dinâmica e ciclo de vida de ponteiros no Heap, descrevendo a causa raiz, os riscos operacionais e apresentando o refatoramento defensivo com Modern C++.",
        "input": "#include <iostream>\n#include <string>\n#include <vector>\n\nstruct UserSession {\n    uint32_t session_id;\n    std::string username;\n    bool is_authenticated;\n    void log_activity() {\n        std::cout << \"[Session \" << session_id << \"] User: \" << username << \" is active.\\n\";\n    }\n};\n\nclass SessionRegistry {\nprivate:\n    std::vector<UserSession*> active_sessions;\n\npublic:\n    void register_session(UserSession* session) {\n        active_sessions.push_back(session);\n    }\n\n    void terminate_session(uint32_t session_id) {\n        for (auto it = active_sessions.begin(); it != active_sessions.end(); ++it) {\n            if ((*it)->session_id == session_id) {\n                delete *it; // Liberação da memória alocada no Heap\n                // Inseguro: ponteiro pendente (dangling pointer) não foi removido do vetor nem aterrado (nullptr)\n                break;\n            }\n        }\n    }\n\n    void audit_all_sessions() {\n        std::cout << \"[*] Executando auditoria periodica de sessoes...\\n\";\n        for (UserSession* session : active_sessions) {\n            // Acesso desreferenciando o ponteiro\n            session->log_activity();\n        }\n    }\n};",
        "reasoning_trace": "<think>\n"
            "**Análise de Superfície**: A classe C++ `SessionRegistry` mantém uma coleção de ponteiros brutos (`UserSession*`) em um vetor dinâmico `std::vector<UserSession*> active_sessions`. O método `terminate_session` busca uma sessão pelo identificador `session_id` e desaloca a memória chamando `delete *it;`. O método `audit_all_sessions` itera sobre todas as instâncias do vetor e invoca o método de membro `session->log_activity()`.\n\n"
            "**Identificação de Pontos de Entrada / Vetores**: O vetor de entrada é o fluxo de eventos da aplicação em rede onde sessões são criadas, destruídas e auditadas concorrentemente ou sequencialmente. Uma solicitação de encerramento de sessão seguida de um ciclo de auditoria aciona o caminho defeituoso.\n\n"
            "**Mapeamento de Fluxo / Mecânica Interna**: Quando `terminate_session(session_id)` localiza o elemento correspondente, a instrução `delete *it;` devolve o bloco de memória associado ao alocador de heap da biblioteca C/C++. No entanto, duas falhas graves ocorrem em sequência: (1) O ponteiro não é removido da lista através do método `active_sessions.erase(it)`; (2) O ponteiro dentro do vetor não é aterrado (`*it = nullptr`), permanecendo como um ponteiro pendente (dangling pointer). Quando o método `audit_all_sessions()` é executado posteriormente, o loop `for (UserSession* session : active_sessions)` acessa o endereço do ponteiro pendente e invoca o método `session->log_activity()`.\n\n"
            "**Detecção de Sinks Perigosos / Falhas de Controle**: O sink perigoso é a desreferenciação `session->log_activity()` (que acessa `session_id` e a string `username`). A desreferenciação de um bloco de heap liberado constitui Comportamento Indefinido (Undefined Behavior - UB).\n\n"
            "**Avaliação de Sanitização e Mitigações**: O código não adota o paradigma RAII (Resource Acquisition Is Initialization), não emprega Smart Pointers (`std::unique_ptr` ou `std::shared_ptr`) e não implementa checagem de estado ou limpeza atômica da estrutura de dados ao desalocar recursos.\n\n"
            "**Conclusão da Vulnerabilidade / Anomalia**: Trata-se de uma vulnerabilidade clássica de Use-After-Free (UAF - CWE-416).\n\n"
            "**Impacto Potencial**: Caso a área de memória do heap previamente liberada seja realocada para outro objeto de dados (via técnicas de Heap Grooming / Heap Spraying), a leitura de `session->username` ou desreferenciação de vtable (caso o objeto possuísse métodos virtuais) permite leitura de dados confidenciais ou Sequestro do Fluxo de Execução (RCE). No cenário comum, causa falha imediata por violação de acesso à memória (Segmentation Fault / Access Violation `0xC0000005`).\n"
            "</think>",
        "final_answer": "FATO: O código é vulnerável a Use-After-Free (UAF - CWE-416). No método `terminate_session()`, a memória do objeto é liberada com `delete *it;`, mas o ponteiro não é removido do vetor `active_sessions` nem definido como nulo. Quando `audit_all_sessions()` é chamado, ele itera sobre ponteiros pendentes (dangling pointers) e desreferencia a memória desalocada em `session->log_activity()`, configurando comportamento indefinido e corrupção de heap.\n\n"
            "DETECÇÃO / TELEMETRIA: Instrumentação com AddressSanitizer (`clang++ -fsanitize=address -g`) que detecta o erro em tempo de execução reportando: `ERROR: AddressSanitizer: heap-use-after-free on address ... READ of size 8`. No Valgrind Memcheck, sinalização imediata de `Invalid read of size 4/8`.\n\n"
            "SUGESTÃO DE CORREÇÃO: Utilizar Modern C++ com smart pointers (`std::shared_ptr` com `std::weak_ptr` ou `std::unique_ptr`) para governança automática do ciclo de vida dos objetos, e garantir que a remoção do container ocorra com o idiom erase-remove:\n"
            "```cpp\n"
            "#include <iostream>\n"
            "#include <string>\n"
            "#include <vector>\n"
            "#include <memory>\n"
            "#include <algorithm>\n\n"
            "struct UserSession {\n"
            "    uint32_t session_id;\n"
            "    std::string username;\n"
            "    bool is_authenticated;\n"
            "    void log_activity() const {\n"
            "        std::cout << \"[Session \" << session_id << \"] User: \" << username << \" is active.\\n\";\n"
            "    }\n"
            "};\n\n"
            "class SessionRegistrySecure {\n"
            "private:\n"
            "    std::vector<std::shared_ptr<UserSession>> active_sessions;\n\n"
            "public:\n"
            "    void register_session(std::shared_ptr<UserSession> session) {\n"
            "        if (session) {\n"
            "            active_sessions.push_back(std::move(session));\n"
            "        }\n"
            "    }\n\n"
            "    void terminate_session(uint32_t session_id) {\n"
            "        active_sessions.erase(\n"
            "            std::remove_if(active_sessions.begin(), active_sessions.end(),\n"
            "                [session_id](const std::shared_ptr<UserSession>& s) {\n"
            "                    return s && s->session_id == session_id;\n"
            "                }),\n"
            "            active_sessions.end()\n"
            "        );\n"
            "    }\n\n"
            "    void audit_all_sessions() const {\n"
            "        std::cout << \"[*] Executando auditoria periodica de sessoes...\\n\";\n"
            "        for (const auto& session : active_sessions) {\n"
            "            if (session) {\n"
            "                session->log_activity();\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "};\n"
            "```"
    })

    # -------------------------------------------------------------------------
    # EXEMPLO 4: Java / AppSec (Insecure Deserialization CWE-502 / JEP 290)
    # -------------------------------------------------------------------------
    samples.append({
        "instruction": "Analise o componente Java abaixo responsável pelo processamento de tokens de autenticação/sessão recebidos pela rede. Avalie o mecanismo de deserialização, identifique os riscos associados ao fluxo de execução da JVM (gadget chains) e proponha as mitigações recomendadas pelas práticas modernas de AppSec (JEP 290).",
        "input": "package com.enterprise.security.auth;\n\nimport java.io.ByteArrayInputStream;\nimport java.io.ObjectInputStream;\nimport java.util.Base64;\nimport javax.servlet.http.HttpServletRequest;\n\npublic class SessionTokenHandler {\n\n    public Object restoreSession(HttpServletRequest request) throws Exception {\n        String sessionCookie = request.getHeader(\"X-Custom-Session\");\n        if (sessionCookie == null || sessionCookie.isEmpty()) {\n            return null;\n        }\n\n        // Decodifica a carga útil enviada pelo cliente em base64\n        byte[] serializedData = Base64.getDecoder().decode(sessionCookie);\n\n        // Deserialização direta usando a API nativa de serialização do Java\n        ByteArrayInputStream bais = new ByteArrayInputStream(serializedData);\n        try (ObjectInputStream ois = new ObjectInputStream(bais)) {\n            // Reconstrução do grafo de objetos em memória\n            return ois.readObject();\n        }\n    }\n}",
        "reasoning_trace": "<think>\n"
            "**Análise de Superfície**: O método Java `restoreSession` extrai o valor de um cabeçalho HTTP customizado (`X-Custom-Session`), decodifica o conteúdo de Base64 em um array de bytes e o submete diretamente a um `ObjectInputStream` para deserialização via `readObject()`.\n\n"
            "**Identificação de Pontos de Entrada / Vetores**: O ponto de entrada é o cabeçalho HTTP `X-Custom-Session`. Os dados são controlados integralmente pelo cliente externo sem autenticação prévia garantida no código exposto.\n\n"
            "**Mapeamento de Fluxo / Mecânica Interna**: O dado decodificado é passado para `ObjectInputStream.readObject()`. O mecanismo nativo de serialização do Java reconstrói objetos invocando automaticamente métodos especiais durante o processo de desserialização (tais como `readObject()`, `readResolve()`, `validateObject()` e manipuladores de proxy dinâmico como `InvocationHandler.invoke()`). Se o classpath da aplicação contiver bibliotecas comuns de terceiros (ex: Apache Commons Collections, Spring, Hibernate, Groovy, etc.), o atacante pode montar um 'Gadget Chain': uma cadeia de invocações reflexivas onde o método de término (sink) atinge classes perigosas como `TemplatesImpl` ou `Runtime.getRuntime().exec()`. Essa execução ocorre *antes* de qualquer cast de tipo ou validação pelo código chamador.\n\n"
            "**Detecção de Sinks Perigosos / Falhas de Controle**: O sink perigoso é a invocação `ois.readObject()` sem nenhuma restrição ou filtro sobre quais classes podem ser instanciadas pela JVM a partir do fluxo de bytes.\n\n"
            "**Avaliação de Sanitização e Mitigações**: Não há nenhuma sanitização ou controle. A mera decodificação Base64 não altera o conteúdo binário malicioso do payload serializado.\n\n"
            "**Conclusão da Vulnerabilidade / Anomalia**: Trata-se de uma vulnerabilidade crítica de Deserialização de Dados Não Confiáveis (CWE-502).\n\n"
            "**Impacto Potencial**: Execução Remota de Código (RCE) completa no contexto da conta de serviço sob a qual o servidor de aplicação/JVM opera, possibilitando roubo de dados, instalação de persistência e movimentação lateral no ambiente de rede corporativo.\n"
            "</think>",
        "final_answer": "FATO: O método `restoreSession` é vulnerável a Deserialização de Dados Não Confiáveis (CWE-502). Ao invocar `ois.readObject()` diretamente sobre bytes decodificados do cabeçalho HTTP `X-Custom-Session`, a JVM permite a instanciação arbitrária de classes e disparo de Gadget Chains presentes no classpath, permitindo Execução Remota de Código (RCE) não autenticada antes que qualquer validação de tipo ocorra.\n\n"
            "DETECÇÃO / TELEMETRIA: Soluções de RASP (Runtime Application Self-Protection) ou agentes APM monitorando a resolução de classes anômalas em `ObjectInputStream.resolveClass()`. No SIEM/EDR, alertas de processos-filhos inesperados spawnados pela JVM (ex: `java.exe` gerando `cmd.exe`, `powershell.exe` ou `/bin/sh`).\n\n"
            "SUGESTÃO DE CORREÇÃO: Substituir a serialização nativa do Java por formatos neutros e seguros como JSON ou Protocol Buffers. Caso o uso de serialização nativa do Java seja estritamente indispensável, aplicar um filtro estrito de classes permitidas (allowlist) via `ObjectInputFilter` (JEP 290):\n"
            "```java\n"
            "package com.enterprise.security.auth;\n\n"
            "import java.io.ByteArrayInputStream;\n"
            "import java.io.ObjectInputFilter;\n"
            "import java.io.ObjectInputStream;\n"
            "import java.util.Base64;\n"
            "import javax.servlet.http.HttpServletRequest;\n\n"
            "public class SessionTokenHandlerSecure {\n\n"
            "    public UserSessionData restoreSession(HttpServletRequest request) throws Exception {\n"
            "        String sessionCookie = request.getHeader(\"X-Custom-Session\");\n"
            "        if (sessionCookie == null || sessionCookie.isEmpty()) {\n"
            "            return null;\n"
            "        }\n\n"
            "        byte[] serializedData = Base64.getDecoder().decode(sessionCookie);\n"
            "        ByteArrayInputStream bais = new ByteArrayInputStream(serializedData);\n\n"
            "        try (ObjectInputStream ois = new ObjectInputStream(bais)) {\n"
            "            // Definicao de filtro restrito baseado na JEP 290: apenas classes autorizadas\n"
            "            ObjectInputFilter filter = ObjectInputFilter.Config.createFilter(\n"
            "                \"com.enterprise.security.auth.UserSessionData;java.lang.String;!*\"\n"
            "            );\n"
            "            ois.setObjectInputFilter(filter);\n\n"
            "            Object obj = ois.readObject();\n"
            "            if (obj instanceof UserSessionData) {\n"
            "                return (UserSessionData) obj;\n"
            "            }\n"
            "            throw new SecurityException(\"Tipo de objeto inesperado na sessao\");\n"
            "        }\n"
            "    }\n"
            "}\n"
            "```"
    })

    # -------------------------------------------------------------------------
    # EXEMPLO 5: Python / Web AppSec (SSRF CWE-918 e IMDSv1/IMDSv2)
    # -------------------------------------------------------------------------
    samples.append({
        "instruction": "Examine a rotina Python de webhook/validação de URLs abaixo. Identifique possíveis falhas de validação de rede (Server-Side Request Forgery - SSRF), avalie os mecanismos de contorno de filtros ingênuos e o impacto em infraestruturas de nuvem (AWS IMDS), propondo a correção robusta com resolução DNS segura.",
        "input": "import urllib.parse\nimport requests\n\ndef fetch_external_resource(target_url: str):\n    # Análise da URL recebida\n    parsed = urllib.parse.urlparse(target_url)\n\n    # Blacklist ingênua para tentar bloquear acessos à rede interna\n    blocked_hosts = [\"localhost\", \"127.0.0.1\", \"0.0.0.0\"]\n    if parsed.hostname in blocked_hosts:\n        raise ValueError(\"Acesso a endereços locais não é permitido.\")\n\n    # Requisição HTTP ao recurso externo\n    response = requests.get(target_url, timeout=5)\n    return response.text",
        "reasoning_trace": "<think>\n"
            "**Análise de Superfície**: A função Python `fetch_external_resource` recebe uma URL `target_url`, analisa seu hostname com `urllib.parse.urlparse` e compara contra uma lista negra (`blocked_hosts`) contendo 'localhost', '127.0.0.1' e '0.0.0.0'. Se passar na verificação, realiza uma requisição HTTP GET usando a biblioteca `requests`.\n\n"
            "**Identificação de Pontos de Entrada / Vetores**: O ponto de entrada é o parâmetro de string `target_url`, comumente exposto em funcionalidades de webhook, download de avatares ou proxies de API.\n\n"
            "**Mapeamento de Fluxo / Mecânica Interna**: A validação apoia-se em uma verificação de string superficial no hostname. Diversos vetores contornam esse filtro trivialmente:\n"
            "1. Endereço do serviço de metadados de nuvem: em ambientes AWS, o IP de link-local `http://169.254.169.254/latest/meta-data/` não está na lista negra;\n"
            "2. Representações alternativas de IP: formatos decimais (`http://2130706433/`), hexadecimais (`http://0x7f000001/`), octais ou IPv6 (`http://[::1]/`);\n"
            "3. Redirecionamento HTTP: o atacante fornece uma URL pública sob seu controle (`http://attacker.com/redirect`), que passa no filtro e emite um `302 Found` apontando para o endpoint interno;\n"
            "4. DNS Rebinding: domínio que resolve para um IP público durante a validação inicial e para um IP privado no momento em que `requests.get` estabelece a conexão socket.\n\n"
            "**Detecção de Sinks Perigosos / Falhas de Controle**: O sink perigoso é `requests.get(target_url)`. Por padrão, `requests` segue redirecionamentos HTTP automaticamente (`allow_redirects=True`) e realiza nova resolução DNS independente.\n\n"
            "**Avaliação de Sanitização e Mitigações**: A sanitização por lista negra (denylist) é conceitualmente falha. Não valida esquemas autorizados (permitindo potencialmente `file://` ou esquemas obscuros), não valida intervalos privados da RFC 1918 (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16) nem blocos link-local (169.254.0.0/16).\n\n"
            "**Conclusão da Vulnerabilidade / Anomalia**: Vulnerabilidade crítica de Server-Side Request Forgery (SSRF - CWE-918).\n\n"
            "**Impacto Potencial**: Em instâncias AWS rodando IMDSv1, o atacante obtém as credenciais temporárias da IAM Role associada à instância (`/latest/meta-data/iam/security-credentials/<role-name>`), permitindo roubo de credenciais da nuvem e escalação de privilégios em toda a conta AWS. Na rede corporativa interna, possibilita port scanning e interação com serviços não autenticados (Redis, Consul, Elasticsearch).\n"
            "</think>",
        "final_answer": "FATO: A função é vulnerável a Server-Side Request Forgery (SSRF - CWE-918). O filtro baseado em denylist estática de strings (`localhost`, `127.0.0.1`) é facilmente contornado por notações de IP alternativas, redirecionamentos HTTP automáticos e, criticamente, pelo endereço de link-local de metadados da AWS (`http://169.254.169.254/latest/meta-data/`), expondo credenciais de IAM da instância.\n\n"
            "DETECÇÃO / TELEMETRIA: VPC Flow Logs monitorando conexões outbound das instâncias de aplicação com destino a `169.254.169.254` na porta 80. Alertas do AWS GuardDuty do tipo `UnauthorizedAccess:EC2/MetadataDNSRebind`. Regras de WAF detectando padrões de IPs de link-local e metadados nos parâmetros da requisição.\n\n"
            "SUGESTÃO DE CORREÇÃO: Implementar validação estrita com resolução prévia de DNS, verificação de todas as famílias de IP privados via módulo `ipaddress`, desativação de redirecionamentos automáticos e, no plano de nuvem, migrar compulsoriamente para o IMDSv2 (`HttpTokens=required`):\n"
            "```python\n"
            "import ipaddress\n"
            "import socket\n"
            "import urllib.parse\n"
            "import requests\n\n"
            "def fetch_external_resource_secure(target_url: str) -> str:\n"
            "    parsed = urllib.parse.urlparse(target_url)\n"
            "    if parsed.scheme not in ('http', 'https'):\n"
            "        raise ValueError(\"Esquema de protocolo nao autorizado\")\n\n"
            "    hostname = parsed.hostname\n"
            "    if not hostname:\n"
            "        raise ValueError(\"Hostname invalido\")\n\n"
            "    # Resolucao DNS de todos os IPs associados ao hostname\n"
            "    addr_info = socket.getaddrinfo(hostname, None)\n"
            "    for entry in addr_info:\n"
            "        ip_str = entry[4][0]\n"
            "        ip_obj = ipaddress.ip_address(ip_str)\n\n"
            "        # Bloqueia IPs privados, loopback, link-local, multicast e reservados\n"
            "        if (\n"
            "            ip_obj.is_private\n"
            "            or ip_obj.is_loopback\n"
            "            or ip_obj.is_link_local\n"
            "            or ip_obj.is_multicast\n"
            "            or ip_obj.is_reserved\n"
            "        ):\n"
            "            raise SecurityError(f\"Acesso a endereco IP restrito negado: {ip_str}\")\n\n"
            "    # Requisicao com desativacao estrita de redirecionamentos\n"
            "    response = requests.get(target_url, timeout=5, allow_redirects=False)\n"
            "    return response.text\n"
            "```"
    })

    # -------------------------------------------------------------------------
    # EXEMPLO 6: Go / Concorrência & AppSec (TOCTOU CWE-367 em Saldo Bancário)
    # -------------------------------------------------------------------------
    samples.append({
        "instruction": "Audite a rotina de processamento financeiro escrita em Go apresentada abaixo. Avalie o tratamento de concorrência, o comportamento das goroutines perante o estado compartilhado da memória e proponha a solução de sincronização atômica adequada.",
        "input": "package main\n\nimport (\n\t\"errors\"\n\t\"fmt\"\n\t\"time\"\n)\n\ntype Account struct {\n\tID      string\n\tBalance float64\n}\n\ntype BankService struct {\n\taccounts map[string]*Account\n}\n\nfunc (s *BankService) Withdraw(accountID string, amount float64) error {\n\taccount, exists := s.accounts[accountID]\n\tif !exists {\n\t\treturn errors.New(\"conta nao encontrada\")\n\t}\n\n\t// Verificação do saldo disponível (Time-of-Check)\n\tif account.Balance < amount {\n\t\treturn errors.New(\"saldo insuficiente\")\n\t}\n\n\t// Simulação de latência de I/O ou processamento assíncrono\n\ttime.Sleep(10 * time.Millisecond)\n\n\t// Atualização do saldo (Time-of-Use)\n\taccount.Balance -= amount\n\tfmt.Printf(\"[Transferencia] Conta %s: debito de %.2f concluido. Novo saldo: %.2f\\n\",\n\t\taccountID, amount, account.Balance)\n\n\treturn nil\n}",
        "reasoning_trace": "<think>\n"
            "**Análise de Superfície**: O serviço Go `BankService` gerencia contas bancárias representadas por ponteiros `*Account` armazenados em um mapa nativo. O método `Withdraw` recebe o identificador da conta e o valor do saque, verifica a existência e o saldo, simula latência com `time.Sleep` e debita o saldo.\n\n"
            "**Identificação de Pontos de Entrada / Vetores**: O ponto de entrada é a função `Withdraw` ao receber requisições disparadas por múltiplos clientes ou requisições concorrentes disparadas pelo mesmo cliente através de APIs HTTP/gRPC simultâneas.\n\n"
            "**Mapeamento de Fluxo / Mecânica Interna**: A arquitetura do Go executa chamadas em goroutines concorrentes sob o controle do scheduler da linguagem. Suponha que o cliente possua saldo de R$ 100,00 e envie duas requisições simultâneas para sacar R$ 100,00: (1) A Goroutine A executa `if account.Balance < amount` (100 < 100 é falso, prossegue) e entra em `time.Sleep`; (2) A Goroutine B é despachada pelo scheduler antes de A concluir; ela lê `account.Balance`, que ainda é 100 (100 < 100 falso, prossegue) e entra em `time.Sleep`; (3) Ambas as goroutines acordam e executam `account.Balance -= amount`. O saldo final torna-se -100,00, gerando um saque total de R$ 200,00 a partir de um saldo real de apenas R$ 100,00.\n\n"
            "**Detecção de Sinks Perigosos / Falhas de Controle**: O sink perigoso é a leitura e escrita simultânea da variável `account.Balance` em memória compartilhada sem barreiras de exclusão mútua (`sync.Mutex`), associada ao acesso concorrente a um `map` de Go sem sincronização (o que pode inclusive deflagrar panic fatal do runtime: `fatal error: concurrent map read and map write`).\n\n"
            "**Avaliação de Sanitização e Mitigações**: Não há mecanismos de exclusão mútua (`sync.Mutex`), tipos atômicos da biblioteca `sync/atomic` nem transações com bloqueio pessimista em banco de dados.\n\n"
            "**Conclusão da Vulnerabilidade / Anomalia**: Vulnerabilidade de Condição de Corrida do tipo Time-of-Check to Time-of-Use (TOCTOU - CWE-367) e Concorrência Insegura de Recursos (CWE-662).\n\n"
            "**Impacto Potencial**: Fraude financeira através de ataque de concorrência (Race Attack / Double Spending), permitindo que atacantes extraiam recursos financeiros ilimitados esvaziando a tesouraria do sistema, além de causar instabilidade e interrupção do serviço da API por crashes no runtime de Go.\n"
            "</think>",
        "final_answer": "FATO: A rotina é vulnerável a Race Condition do tipo TOCTOU (Time-of-Check to Time-of-Use / CWE-367) e acesso concorrente não sincronizado à memória. Devido à ausência de exclusão mútua entre a checagem de saldo e a sua dedução, múltiplas goroutines concorrentes podem passar pela validação de saldo antes que a primeira registre o débito, permitindo saques duplicados (Double Spending) e saldo negativo, além de risco de crash no runtime do Go por leituras e escritas concorrentes no mapa.\n\n"
            "DETECÇÃO / TELEMETRIA: Compilação e execução de testes de estresse com o Race Detector nativo do Go ativado (`go test -race` ou `go run -race`), disparando mensagens como: `WARNING: DATA RACE Write at ... by goroutine ... Previous read at ... by goroutine ...`. Em produção, monitoramento de discrepâncias na conciliação contábil entre eventos de débito e saldos consolidados.\n\n"
            "SUGESTÃO DE CORREÇÃO: Implementar sincronização com `sync.Mutex` granular por conta ou a nível de serviço para isolar a seção crítica, ou aplicar bloqueio em transações de banco de dados (`SELECT ... FOR UPDATE`):\n"
            "```go\n"
            "package main\n\n"
            "import (\n"
            "\t\"errors\"\n"
            "\t\"fmt\"\n"
            "\t\"sync\"\n"
            ")\n\n"
            "type AccountSecure struct {\n"
            "\tID      string\n"
            "\tBalance float64\n"
            "\tmu      sync.Mutex // Mutex para protecao do estado interno da conta\n"
            "}\n\n"
            "type BankServiceSecure struct {\n"
            "\tsync.RWMutex\n"
            "\taccounts map[string]*AccountSecure\n"
            "}\n\n"
            "func (s *BankServiceSecure) Withdraw(accountID string, amount float64) error {\n"
            "\tif amount <= 0 {\n"
            "\t\treturn errors.New(\"valor de saque deve ser positivo\")\n"
            "\t}\n\n"
            "\ts.RLock()\n"
            "\taccount, exists := s.accounts[accountID]\n"
            "\ts.RUnlock()\n\n"
            "\tif !exists {\n"
            "\t\treturn errors.New(\"conta nao encontrada\")\n"
            "\t}\n\n"
            "\t// Bloqueio atomico da conta durante a verificacao e o debito\n"
            "\taccount.mu.Lock()\n"
            "\tdefer account.mu.Unlock()\n\n"
            "\tif account.Balance < amount {\n"
            "\t\treturn errors.New(\"saldo insuficiente\")\n"
            "\t}\n\n"
            "\taccount.Balance -= amount\n"
            "\tfmt.Printf(\"[Transferencia Segura] Conta %s: debito de %.2f concluido. Novo saldo: %.2f\\n\",\n"
            "\t\taccountID, amount, account.Balance)\n\n"
            "\treturn nil\n"
            "}\n"
            "```"
    })

    # -------------------------------------------------------------------------
    # EXEMPLO 7: Active Directory / Blue Team (Kerberos Constrained Delegation)
    # -------------------------------------------------------------------------
    samples.append({
        "instruction": "Analise a configuração de Active Directory e a telemetria SIEM abaixo envolvendo delegação Kerberos restrita. Explique como as extensões S4U (Service-for-User) podem ser abusadas por um atacante com posse da conta de serviço, detalhe os Event IDs de detecção e descreva as medidas de contenção e hardening.",
        "input": "[AD Object Configuration: Service Account]\nsAMAccountName: svc_webportal\nuserAccountControl: 0x1000 (WORKSTATION_TRUST_ACCOUNT) / TRUSTED_TO_AUTHENTICATE_FOR_DELEGATION\nmsDS-AllowedToDelegateTo:\n    - cifs/fileserver.corp.internal\n    - cifs/fileserver\n\n[SIEM Alert - Windows Security Event Log]\nEvent ID: 4769 (A Kerberos service ticket was requested)\nService Name: cifs/fileserver.corp.internal\nTargetUserName: svc_webportal$@CORP.INTERNAL\nServiceName: fileserver$\nTicket Options: 0x40810000\nTicket Encryption Type: 0x12 (AES256)\nTransiting Services: svc_webportal",
        "reasoning_trace": "<think>\n"
            "**Análise de Superfície**: Um objeto de conta de serviço no Active Directory (`svc_webportal`) está configurado com a flag `TRUSTED_TO_AUTHENTICATE_FOR_DELEGATION` (Protocol Transition ativo) e possui o atributo `msDS-AllowedToDelegateTo` apontando para o serviço CIFS em `fileserver.corp.internal`. O alerta do SIEM exibe uma solicitação de TGS (Event ID 4769) com serviços de trânsito intermediário.\n\n"
            "**Identificação de Pontos de Entrada / Vetores**: O ponto de entrada ocorre caso a conta de serviço `svc_webportal` tenha suas credenciais, hash NTLM ou TGT comprometidos por um invasor (por exemplo, via Kerberoasting, extração de LSASS ou dump de memória do servidor web).\n\n"
            "**Mapeamento de Fluxo / Mecânica Interna**: As extensões Kerberos da Microsoft para delegação restrita consistem em: (1) S4U2self (Service-for-User-to-Self): permite que a conta de serviço solicite um TGS para si mesma em nome de *qualquer usuário do domínio*, sem que esse usuário precise autenticar ou fornecer senha; (2) S4U2proxy (Service-for-User-to-Proxy): permite que a conta use o ticket obtido via S4U2self para solicitar um TGS para os SPNs definidos em `msDS-AllowedToDelegateTo` (neste caso, `cifs/fileserver`). Com o privilégio `TRUSTED_TO_AUTHENTICATE_FOR_DELEGATION` (Protocol Transition), o invasor não precisa de um TGS prévio encaminhável emitido pelo usuário. O atacante forja um ticket personificando um Domain Admin (ex: `Administrator`) e solicita acesso ao serviço CIFS do fileserver alvo.\n\n"
            "**Detecção de Sinks Perigosos / Falhas de Controle**: O sink é o Key Distribution Center (KDC) aceitando a transição de protocolo e emitindo um TGS com privilégios de administrador sem que as contas de alto privilégio tenham sido protegidas contra delegação.\n\n"
            "**Avaliação de Sanitização e Mitigações**: A configuração atual permite personificação irrestrita de identidade porque os administradores do domínio não foram incluídos no grupo `Protected Users` nem tiveram o atributo `Account is sensitive and cannot be delegated` habilitado em seus perfis.\n\n"
            "**Conclusão da Vulnerabilidade / Anomalia**: Abuso de Kerberos Constrained Delegation com Protocol Transition (MITRE ATT&CK T1558 - Steal or Forge Kerberos Tickets).\n\n"
            "**Impacto Potencial**: Escalação de privilégios e movimentação lateral instantânea para o servidor de arquivos (`fileserver`), permitindo leitura de dados confidenciais de rede ou comprometimento total do sistema operacional host via SMB/RPC caso o atacante delegue para serviços administrativos como `HOST` ou `RPCSS`.\n"
            "</think>",
        "final_answer": "FATO: A conta `svc_webportal` está configurada com Kerberos Constrained Delegation com Protocol Transition habilitado (`TRUSTED_TO_AUTHENTICATE_FOR_DELEGATION`). Essa configuração permite que qualquer invasor de posse das credenciais da conta de serviço utilize a extensão Kerberos S4U2self para gerar tickets em nome de qualquer identidade do domínio (incluindo Domain Admins) e usá-los via S4U2proxy para acessar o serviço CIFS no servidor de destino especificado em `msDS-AllowedToDelegateTo`.\n\n"
            "DETECÇÃO / TELEMETRIA: Monitoramento de Windows Security Event ID 4769 (\"A Kerberos service ticket was requested\") correlacionando eventos onde o campo `Transiting Services` esteja preenchido por contas de serviço solicitando tickets para contas de alta gerência ou administradores. Consulta KQL para Microsoft Sentinel:\n"
            "```kql\n"
            "SecurityEvent\n"
            "| where EventID == 4769\n"
            "| extend TicketOptions = tostring(extract(@\"Ticket Options:\\s+(0x[0-9a-fA-F]+)\", 1, EventData))\n"
            "| extend TransitingServices = tostring(extract(@\"Transiting Services:\\s+([^\\r\\n]+)\", 1, EventData))\n"
            "| where isnotempty(TransitingServices) and TransitingServices != \"-\"\n"
            "| project TimeGenerated, TargetAccount, ServiceName, TransitingServices, TicketOptions, IpAddress\n"
            "```\n\n"
            "SUGESTÃO DE CORREÇÃO: 1) Adicionar todos os usuários administrativos aos grupos `Protected Users` (bloqueia o uso de credenciais em delegação Kerberos); 2) Marcar o atributo `Account is sensitive and cannot be delegated` (USER_NOT_DELEGATED) nos objetos de usuários de alta hierarquia no AD; 3) Migrar para Resource-Based Constrained Delegation (RBCD) onde o controle de quem pode delegar é determinado pelo recurso de destino, desativando a transição de protocolo irrestrita na conta de serviço."
    })

    # -------------------------------------------------------------------------
    # EXEMPLO 8: Windows Internals / Telemetria EDR (User-Mode Hooking vs ETW-Ti)
    # -------------------------------------------------------------------------
    samples.append({
        "instruction": "Avalie a arquitetura de monitoramento de processos em endpoints Windows descrita a seguir. Compare a eficácia da detecção via User-Mode API Hooking (`ntdll.dll`) em relação a Kernel Callbacks e ETW-Ti (`Microsoft-Windows-Threat-Intelligence`) na identificação de alocações de memória executável (`PAGE_EXECUTE_READWRITE`), detalhando técnicas de evasão e telemetria defensiva resiliente.",
        "input": "// Trecho de loader em C executado por um agente malicioso\n#include <windows.h>\n\nvoid AllocatePayload() {\n    LPVOID pMemory = VirtualAlloc(\n        NULL,\n        4096,\n        MEM_COMMIT | MEM_RESERVE,\n        PAGE_EXECUTE_READWRITE // RWX alocado diretamente\n    );\n\n    // Em seguida, o loader efetua a restauração da ntdll.dll lendo os bytes\n    // originais de disco (\\KnownDlls\\ntdll.dll) para sobrescrever os jmps (unhooking)\n    // ou invoca a syscall NtAllocateVirtualMemory diretamente via assembly (direct syscalls).\n}",
        "reasoning_trace": "<think>\n"
            "**Análise de Superfície**: Um loader executando em espaço de usuário (Ring 3) no Windows aloca uma região de memória virtual com proteção `PAGE_EXECUTE_READWRITE` (RWX) e descreve o uso de técnicas de evasão de EDR como DLL unhooking e direct syscalls.\n\n"
            "**Identificação de Pontos de Entrada / Vetores**: O vetor é a execução de um binário ou script não assinado no endpoint corporativo, buscando carregar um implante ou beacon C2 em memória sem gerar arquivos em disco (fileless).\n\n"
            "**Mapeamento de Fluxo / Mecânica Interna**: Grande parte dos agentes de EDR tradicionais monitoram processos aplicando Inline Hooking na `ntdll.dll` carregada no espaço de endereçamento do processo em Ring 3 (inserindo uma instrução `JMP` no início de funções como `NtAllocateVirtualMemory`). Como o código malicioso opera com os mesmos privilégios do espaço de memória do processo em Ring 3, ele contorna os hooks através de: (1) Unhooking: mapeia uma cópia limpa da seção `.text` da `ntdll.dll` a partir do disco e restaura os bytes originais, apagando os `JMP` do EDR; (2) Direct Syscalls: o malware extrai o System Service Number (SSN) e emite a instrução `syscall` diretamente no assembly, ignorando completamente as funções da `ntdll.dll`. No entanto, quando a instrução `syscall` transita para o Kernel (Ring 0), os mecanismos de EDR baseados em kernel operam sem interferência: ETW-Ti (`Microsoft-Windows-Threat-Intelligence`) intercepta a execução da syscall no kernel e gera eventos de telemetria independentemente de qualquer manipulação na memória do processo de usuário.\n\n"
            "**Detecção de Sinks Perigosos / Falhas de Controle**: A falha fundamental de controle é a dependência de sensores em User-Mode (Ring 3), que compartilham fronteiras de confiança com o código do invasor.\n\n"
            "**Avaliação de Sanitização e Mitigações**: Hooks em User-Mode não fornecem garantias de integridade nem não-repúdio. Apenas verificações no Kernel (Ring 0) e telemetria de hardware (LBR, Intel CET) são resilientes a evasões de Ring 3.\n\n"
            "**Conclusão da Vulnerabilidade / Anomalia**: Evasão de Defesas via Direct Syscalls e DLL Unhooking (MITRE ATT&CK T1055 e T1562.001).\n\n"
            "**Impacto Potencial**: Execução furtiva de código hostil em memória, cegueira analítica de soluções antivírus/EDR baseadas exclusivamente em user-mode hooks e execução de shellcodes sem detecção prévia.\n"
            "</think>",
        "final_answer": "FATO: Soluções defensivas que dependem exclusivamente de User-Mode Hooking na `ntdll.dll` são facilmente neutralizadas por invasores em Ring 3 através de técnicas de Unhooking de DLL (sobrescrita da seção `.text`) e Direct Syscalls (invocação direta da instrução `syscall` via assembly). A alocação de páginas `PAGE_EXECUTE_READWRITE` (RWX) é um indicador crítico de injeção de código que deve ser capturado na camada de Kernel.\n\n"
            "DETECÇÃO / TELEMETRIA: Utilização do provedor de kernel ETW-Ti (`Microsoft-Windows-Threat-Intelligence` GUID `{F4E1897C-BB5D-5668-AC15-64A7041B19DE}`), consumível por drivers ELAM. Os eventos `KERNEL_THREATINT_TASK_ALLOCVM_LOCAL` e `ALLOCVM_REMOTE` registram a alocação de memória executável diretamente da transição de kernel com o campo `Protection` igual a `0x40` (`PAGE_EXECUTE_READWRITE`). Além disso, registro de callbacks de kernel via `ObRegisterCallbacks` monitorando criação de handles com direitos `PROCESS_VM_WRITE | PROCESS_VM_OPERATION`.\n\n"
            "SUGESTÃO DE CORREÇÃO: 1) Ativar a política do Exploit Guard / Arbitrary Code Guard (ACG), que impede que threads criem ou modifiquem páginas de memória para torná-las executáveis (bloqueia transições de RW para RX/RWX); 2) Habilitar mitigação de Code Integrity Guard (CIG) para impedir o carregamento de DLLs não assinadas pela Microsoft; 3) Garantir que a arquitetura do EDR consuma primordialmente Kernel Callbacks (`PsSetCreateProcessNotifyRoutineEx`, `PsSetCreateThreadNotifyRoutine`) e telemetria ETW-Ti."
    })

    # -------------------------------------------------------------------------
    # EXEMPLO 9: Forense / Sysmon & Sigma (Process Hollowing / Injeção)
    # -------------------------------------------------------------------------
    samples.append({
        "instruction": "Analise a cadeia de eventos de telemetria forense do Windows Sysmon abaixo. Correlacione os eventos para identificar a técnica de injeção de processo utilizada, explique a anomalia operacional e forneça uma regra de detecção Sigma completa em YAML.",
        "input": "[Sysmon Event ID 1: Process Creation]\nUtcTime: 2026-10-19 14:22:01.120\nProcessGuid: {a1b2c3d4-0001-0000-0010-000000000000}\nProcessId: 4812\nImage: C:\\Windows\\System32\\svchost.exe\nCommandLine: \"C:\\Windows\\System32\\svchost.exe\" -k netsvcs\nParentProcessGuid: {a1b2c3d4-0000-0000-0010-000000000000}\nParentProcessId: 3240\nParentImage: C:\\Users\\john\\AppData\\Local\\Temp\\invoice_scan.exe\nParentCommandLine: \"C:\\Users\\john\\AppData\\Local\\Temp\\invoice_scan.exe\"\n\n[Sysmon Event ID 25: ProcessTampering (Process Hollowing / Image replacement)]\nUtcTime: 2026-10-19 14:22:01.350\nProcessGuid: {a1b2c3d4-0001-0000-0010-000000000000}\nProcessId: 4812\nImage: C:\\Windows\\System32\\svchost.exe\nType: Image is replaced\nTampered: true\n\n[Sysmon Event ID 8: CreateRemoteThread]\nUtcTime: 2026-10-19 14:22:01.410\nSourceProcessId: 3240\nSourceImage: C:\\Users\\john\\AppData\\Local\\Temp\\invoice_scan.exe\nTargetProcessId: 4812\nTargetImage: C:\\Windows\\System32\\svchost.exe\nStartAddress: 0x00007FF63B121000\nStartFunction: -",
        "reasoning_trace": "<think>\n"
            "**Análise de Superfície**: Os logs do Sysmon registram a criação de um processo legítimo do Windows (`svchost.exe`, PID 4812) gerado por um binário suspeito localizado na pasta temporária do usuário (`invoice_scan.exe`, PID 3240), seguido por um evento de adulteração de processo (Event ID 25) e criação de thread remota (Event ID 8).\n\n"
            "**Identificação de Pontos de Entrada / Vetores**: O vetor inicial é a execução de um binário malicioso em espaço de usuário (`invoice_scan.exe`), característico de anexo recebido em campanha de phishing.\n\n"
            "**Mapeamento de Fluxo / Mecânica Interna**: A correlação dos eventos expõe a mecânica de Process Hollowing (RunPE): (1) O processo malicioso cria uma instância legítima de `svchost.exe` em estado suspenso (`CREATE_SUSPENDED`). Em um ambiente Windows íntegro, instâncias de `svchost.exe` são exclusivamente filhas do processo de gerenciamento de serviços do sistema (`services.exe`). Ter como processo-pai um executável em `AppData\\Local\\Temp` é uma quebra severa da árvore de processos; (2) O invasor desmapeia a imagem original de memória com `NtUnmapViewOfSection` ou aloca novas páginas e sobrescreve o ponto de entrada (PE) com código malicioso. O Sysmon captura isso com precisão através do Event ID 25 (`Type: Image is replaced`); (3) O invasor cria uma thread remota (Event ID 8) ou retoma a thread principal suspensa para executar o payload injetado, com endereço de início anômalo não associado a nenhuma DLL oficial.\n\n"
            "**Detecção de Sinks Perigosos / Falhas de Controle**: O sink operacional é a injeção de código em processos protegidos do sistema operacional sem bloqueio de políticas de integridade de processo.\n\n"
            "**Avaliação de Sanitização e Mitigações**: O invasor tenta se ocultar atrás de um processo de sistema (`svchost.exe`) para mascarar conexões de rede em ferramentas convencionais de monitoramento.\n\n"
            "**Conclusão da Vulnerabilidade / Anomalia**: Process Hollowing / Process Injection (MITRE ATT&CK T1055.012).\n\n"
            "**Impacto Potencial**: Execução de código furtivo sob a identidade de processos do sistema Windows, evasão de firewalls locais e persistência com privilégios ampliados.\n"
            "</think>",
        "final_answer": "FATO: Os logs evidenciam a técnica de Process Hollowing (MITRE ATT&CK T1055.012). O executável não confiável `invoice_scan.exe` gerou uma instância de `svchost.exe` em estado suspenso (anomalia de linhagem de processo, pois `svchost.exe` deve ter como pai exclusivo `services.exe`), substituiu a imagem do binário na memória (Sysmon Event ID 25: ProcessTampering) e iniciou a execução do payload via thread remota (Sysmon Event ID 8).\n\n"
            "DETECÇÃO / TELEMETRIA: Regra Sigma completa em YAML para detecção de Process Tampering e criação anômala de instâncias de `svchost.exe`:\n"
            "```yaml\n"
            "title: Process Hollowing em Processo Svchost via Sysmon\n"
            "id: 9b2d3e4f-1234-5678-abcd-ef0123456789\n"
            "status: experimental\n"
            "description: Detecta a técnica de Process Hollowing ou substituição de imagem em svchost.exe gerado por pais anômalos.\n"
            "references:\n"
            "    - https://attack.mitre.org/techniques/T1055/012/\n"
            "author: CyberRott Defesa\n"
            "date: 2026-10-19\n"
            "logsource:\n"
            "    product: windows\n"
            "    service: sysmon\n"
            "detection:\n"
            "    selection_tampering:\n"
            "        EventID: 25\n"
            "        Image|endswith: '\\svchost.exe'\n"
            "        Type: 'Image is replaced'\n"
            "    selection_creation:\n"
            "        EventID: 1\n"
            "        Image|endswith: '\\svchost.exe'\n"
            "    filter_legit_parent:\n"
            "        ParentImage|endswith: '\\services.exe'\n"
            "    condition: selection_tampering or (selection_creation and not filter_legit_parent)\n"
            "fields:\n"
            "    - ProcessId\n"
            "    - Image\n"
            "    - ParentImage\n"
            "    - CommandLine\n"
            "falsepositives:\n"
            "    - Desconhecido em ambientes corporativos padrão\n"
            "level: critical\n"
            "tags:\n"
            "    - attack.defense_evasion\n"
            "    - attack.privilege_escalation\n"
            "    - attack.t1055.012\n"
            "```\n\n"
            "SUGESTÃO DE CORREÇÃO: 1) Isolar o endpoint imediatamente via console de EDR; 2) Coletar dump de memória do processo PID 4812 para extração de strings e do payload PE injetado; 3) Encerrar os processos maliciosos; 4) Implementar políticas de AppLocker / Software Restriction Policies (SRP) proibindo a execução de binários a partir dos diretórios `%AppData%` e `%Temp%`."
    })

    # -------------------------------------------------------------------------
    # EXEMPLO 10: Engenharia Reversa Defensiva / Binários (Format String CWE-134)
    # -------------------------------------------------------------------------
    samples.append({
        "instruction": "Examine o código em C descompilado obtido na análise reversa de um serviço de rede legado. Identifique a falha na chamada de impressão de strings, analise a mecânica de leitura e escrita arbitrária na memória do processo via especificadores de formato e forneça as correções de código e flags do compilador.",
        "input": "// Trecho descompilado de daemon de auditoria de rede (x86_64 ELF)\n#include <stdio.h>\n#include <unistd.h>\n#include <string.h>\n\nvoid log_client_message(int client_fd) {\n    char user_buffer[512];\n    memset(user_buffer, 0, sizeof(user_buffer));\n\n    // Leitura dos dados brutos recebidos da conexao\n    ssize_t bytes_received = read(client_fd, user_buffer, sizeof(user_buffer) - 1);\n    if (bytes_received <= 0) {\n        return;\n    }\n\n    // Inseguro: buffer controlado pelo usuario passado diretamente como formato\n    printf(user_buffer);\n}",
        "reasoning_trace": "<think>\n"
            "**Análise de Superfície**: O código C descompilado lê até 511 bytes de um socket de rede (`client_fd`) para uma variável de stack `user_buffer` e imprime o conteúdo chamando diretamente `printf(user_buffer)` sem fornecer uma string de formato literal.\n\n"
            "**Identificação de Pontos de Entrada / Vetores**: O ponto de entrada é a chamada `read(client_fd, ...)` que recebe dados arbitrários de clientes da rede sem restrição.\n\n"
            "**Mapeamento de Fluxo / Mecânica Interna**: A função `printf` possui a assinatura `int printf(const char *format, ...)`. Em chamadas de convenção x86_64 (System V AMD64 ABI), os argumentos são passados sequencialmente nos registradores RDI, RSI, RDX, RCX, R8, R9 e na stack. Quando `printf` interpreta a string e encontra especificadores de formato (como `%p`, `%x`, `%s`, `%n`): (1) Leitura arbitrária (Information Disclosure): o uso repetido de `%p` ou `%x` lê valores armazenados nos registradores e na pilha, permitindo vazar canaries de stack, ponteiros da libc e endereços do binário (bypassing ASLR); (2) Escrita arbitrária (Arbitrary Memory Write): o especificador `%n` instrui o `printf` a escrever o número de caracteres impressos até o momento no endereço de memória apontado pelo argumento correspondente na stack. Utilizando especificadores posicionais (ex: `%12$n`) combinados com acolchoamento de largura de campo, o invasor pode sobrescrever ponteiros de função salvos na stack ou entradas na Global Offset Table (GOT).\n\n"
            "**Detecção de Sinks Perigosos / Falhas de Controle**: O sink perigoso é a invocação `printf(user_buffer)` onde o argumento de formatação é inteiramente fornecido pelo cliente externo.\n\n"
            "**Avaliação de Sanitização e Mitigações**: Não há validação da presença de caracteres de formatação (`%`), nem utilização de especificador de formato estático constante.\n\n"
            "**Conclusão da Vulnerabilidade / Anomalia**: Vulnerabilidade crítica de Format String (CWE-134 - Use of Externally-Controlled Format String).\n\n"
            "**Impacto Potencial**: Vazamento de memória sensível (canaries, chaves criptográficas e layouts de memória quebrando ASLR) e Execução Remota de Código (RCE) via sobrescrita arbitrária de ponteiros de retorno ou entradas da GOT.\n\n"
            "</think>",
        "final_answer": "FATO: O código é vulnerável a Format String (CWE-134). Ao passar `user_buffer` diretamente como primeiro argumento para `printf()`, o programa permite que um invasor remoto envie especificadores de formatação como `%p` e `%x` para vazar dados da stack e quebrar o ASLR (Information Disclosure), ou utilizar o especificador `%n` para realizar escrita arbitrária em ponteiros da memória, alcançando Execução Remota de Código (RCE).\n\n"
            "DETECÇÃO / TELEMETRIA: Alertas de compilação automáticos gerados por `-Wformat -Wformat-security`. Em execução, detecção por auditd monitorando falhas anômalas com sinal SIGSEGV provocadas por tentativas de desreferenciação de ponteiros inválidos no daemon.\n\n"
            "SUGESTÃO DE CORREÇÃO: No código-fonte C, fornecer uma string de formato literal estática `\"%s\"` ou utilizar funções sem interpretação de formato como `fputs()`:\n"
            "```c\n"
            "// Correcao segura no codigo C\n"
            "printf(\"%s\", user_buffer);\n"
            "// Ou alternativamente:\n"
            "// fputs(user_buffer, stdout);\n"
            "```\n"
            "No pipeline de compilação defensiva, impor as flags de segurança obrigatórias no GCC/Clang:\n"
            "```bash\n"
            "gcc -O2 -Wformat -Wformat-security -Werror=format-security -D_FORTIFY_SOURCE=2 -fstack-protector-strong -Wl,-z,relro,-z,now -o daemon_secure daemon.c\n"
            "```"
    })

    # -------------------------------------------------------------------------
    # EXEMPLO 11: Smart Contracts / Solidity (Reentrancy CWE-841 e CEI Pattern)
    # -------------------------------------------------------------------------
    samples.append({
        "instruction": "Analise o contrato inteligente Ethereum escrito em Solidity abaixo. Identifique a vulnerabilidade no fluxo de transferência de fundos, explique a mecânica de execução do opcode CALL e a reentrância de contratos externos, apresentando a correção com o padrão Checks-Effects-Interactions e ReentrancyGuard.",
        "input": "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0;\n\ncontract EtherVault {\n    mapping(address => uint256) public userBalances;\n\n    function deposit() external payable {\n        require(msg.value > 0, \"Valor deve ser maior que zero\");\n        userBalances[msg.sender] += msg.value;\n    }\n\n    function withdraw(uint256 amount) external {\n        require(userBalances[msg.sender] >= amount, \"Saldo insuficiente\");\n\n        // Transferencia externa via chamada de baixo nivel enviando ether\n        (bool success, ) = msg.sender.call{value: amount}(\"\");\n        require(success, \"Falha na transferencia\");\n\n        // Atualizacao de estado (saldo) apos a chamada externa\n        userBalances[msg.sender] -= amount;\n    }\n}",
        "reasoning_trace": "<think>\n"
            "**Análise de Superfície**: O contrato inteligente Solidity `EtherVault` permite que usuários depositem Ether e posteriormente realizem saques através do método `withdraw(uint256 amount)`. O método valida o saldo, executa uma transferência de baixo nível chamando `msg.sender.call{value: amount}(\"\")` e depois atualiza o estado deduzindo o saldo em `userBalances[msg.sender]`.\n\n"
            "**Identificação de Pontos de Entrada / Vetores**: O ponto de entrada é a função pública `withdraw(uint256 amount)`, aberta para invocação por qualquer endereço de carteira (EOA) ou contrato inteligente na rede blockchain.\n\n"
            "**Mapeamento de Fluxo / Mecânica Interna**: A ordem de execução viola a integridade temporal de estado na EVM: (1) Checagem: `require(userBalances[msg.sender] >= amount)`; (2) Interação externa: `msg.sender.call{value: amount}(\"\")`; (3) Efeito de estado: `userBalances[msg.sender] -= amount`. Quando o opcode `CALL` transfere Ether para outro contrato inteligente, a EVM transfere o controle de execução para a função `fallback()` ou `receive()` do contrato de destino. Um contrato atacante pode chamar novamente `EtherVault.withdraw(amount)` dentro de seu `receive()`. Como o estado de saldo ainda não foi decrementado na chamada pai, a verificação de saldo `require` passa novamente com sucesso, repetindo o saque de fundos recursivamente até esgotar o balanço de Ether de todo o contrato ou atingir o limite de gás da transação.\n\n"
            "**Detecção de Sinks Perigosos / Falhas de Controle**: O sink perigoso é a invocação de chamada externa `msg.sender.call{value: amount}(\"\")` antes de consolidar a alteração do saldo interno do usuário.\n\n"
            "**Avaliação de Sanitização e Mitigações**: O contrato não utiliza travas de reentrância (ReentrancyGuard) e não aplica o princípio de Checks-Effects-Interactions (CEI).\n\n"
            "**Conclusão da Vulnerabilidade / Anomalia**: Vulnerabilidade crítica de Reentrancy (CWE-841 - Improper Enforcement of Behavioral Workflow).\n\n"
            "**Impacto Potencial**: Drenagem catastrófica e irreversível de todos os fundos de Ether mantidos sob custódia do contrato inteligente por invasores (análogo ao incidente da The DAO).\n\n"
            "</think>",
        "final_answer": "FATO: O contrato `EtherVault` é vulnerável a Reentrancy (CWE-841). Na função `withdraw()`, a transferência externa de fundos (`msg.sender.call{value: amount}(\"\")`) é realizada antes de deduzir o saldo do chamador no mapeamento `userBalances`. Se o chamador for um contrato inteligente, sua função `receive()` ou `fallback()` pode invocar `withdraw()` recursivamente antes da atualização do estado, esvaziando o cofre de Ether.\n\n"
            "DETECÇÃO / TELEMETRIA: Ferramentas de análise estática como Slither reportando a falha com o detector `reentrancy-eth`. Em tempo de execução, monitoramento de transações em nós RPC detectando chamadas internas com profundidade anômala (`call depth > 5`) e taxa de transferência repetida de valor.\n\n"
            "SUGESTÃO DE CORREÇÃO: Aplicar o padrão Checks-Effects-Interactions (CEI) alterando o estado antes de interações externas, e integrar o modificador `nonReentrant` da biblioteca OpenZeppelin `ReentrancyGuard`:\n"
            "```solidity\n"
            "// SPDX-License-Identifier: MIT\n"
            "pragma solidity ^0.8.20;\n\n"
            "import \"@openzeppelin/contracts/utils/ReentrancyGuard.sol\";\n\n"
            "contract EtherVaultSecure is ReentrancyGuard {\n"
            "    mapping(address => uint256) public userBalances;\n\n"
            "    function deposit() external payable {\n"
            "        require(msg.value > 0, \"Valor deve ser maior que zero\");\n"
            "        userBalances[msg.sender] += msg.value;\n"
            "    }\n\n"
            "    function withdraw(uint256 amount) external nonReentrant {\n"
            "        // 1. Checks\n"
            "        require(userBalances[msg.sender] >= amount, \"Saldo insuficiente\");\n\n"
            "        // 2. Effects (atualizacao de estado antes da chamada externa)\n"
            "        userBalances[msg.sender] -= amount;\n\n"
            "        // 3. Interactions\n"
            "        (bool success, ) = msg.sender.call{value: amount}(\"\");\n"
            "        require(success, \"Falha na transferencia\");\n"
            "    }\n"
            "}\n"
            "```"
    })

    # -------------------------------------------------------------------------
    # EXEMPLO 12: Segurança de Rede / Forense DNS (DNS Tunneling)
    # -------------------------------------------------------------------------
    samples.append({
        "instruction": "Avalie a amostra de logs de tráfego de DNS capturados em um gateway corporativo. Analise as características estatísticas e sintáticas dos registros, determine se configuram tunelamento de DNS (C2 / Exfiltração), e construa a query KQL para detecção contínua e as medidas de bloqueio no firewall de DNS (RPZ).",
        "input": "[Gateway DNS Logs (Zeek dns.log format)]\ntimestamp               query_type  query_name                                          response_code   bytes_answer\n2026-10-19T10:14:02Z    TXT         a8f19c3b4e2d1a0f9b8c7d6e5f.c2.tunnelcorp.xyz         NOERROR         256\n2026-10-19T10:14:03Z    TXT         9e8d7c6b5a4f3e2d1c0b9a8f7e.c2.tunnelcorp.xyz         NOERROR         256\n2026-10-19T10:14:04Z    TXT         4b3c2d1e0f9a8b7c6d5e4f3a2b.c2.tunnelcorp.xyz         NOERROR         256\n2026-10-19T10:14:05Z    A           f0e1d2c3b4a5968778695a4b3c.c2.tunnelcorp.xyz         NOERROR         4\n2026-10-19T10:14:06Z    TXT         1a2b3c4d5e6f7a8b9c0d1e2f3a.c2.tunnelcorp.xyz         NOERROR         256\n... (1.200 queries similares nos ultimos 10 minutos para subdominios sob c2.tunnelcorp.xyz)",
        "reasoning_trace": "<think>\n"
            "**Análise de Superfície**: Os registros de log do sensor Zeek exibem um fluxo massivo de consultas de DNS com alta frequência (> 1.200 consultas em 10 minutos) direcionadas a subdomínios sob o domínio apex `tunnelcorp.xyz`, com predominância incomum de registros do tipo `TXT` e tamanho uniforme de resposta de 256 bytes.\n\n"
            "**Identificação de Pontos de Entrada / Vetores**: O vetor é um endpoint interno comprometido que utiliza o canal de resolução de nomes de DNS do resolvedor corporativo como proxy de saída para contornar restrições de firewall de borda.\n\n"
            "**Mapeamento de Fluxo / Mecânica Interna**: A análise forense do tráfego aponta as marcas inequívocas de DNS Tunneling (MITRE ATT&CK T1071.004): (1) Alta Entropia de Shannon no label do subdomínio: sequências pseudo-aleatórias hexadecimais codificando dados exfiltrados (exfiltração outbound); (2) Relação desproporcional de tipos de consulta: em estações de trabalho normais, consultas `A` e `AAAA` compõem mais de 95% do volume, enquanto consultas massivas de registros `TXT` e `NULL` são utilizadas para retorno de payloads e comandos C2 (inbound canalizado); (3) Alta cardinalidade de FQDNs únicos resolvidos sob um único domínio apex em curtos intervalos de tempo.\n\n"
            "**Detecção de Sinks Perigosos / Falhas de Controle**: O sink de controle é a ausência de inspeção estatística e comportamental no resolvedor recursivo de DNS corporativo e liberação irrestrita de consultas para domínios de reputação recém-registrada (NRD).\n\n"
            "**Avaliação de Sanitização e Mitigações**: Firewalls tradicionais de camada 4 liberam a porta UDP 53 sem validação de conteúdo dos pacotes nem análise de entropia de strings.\n\n"
            "**Conclusão da Vulnerabilidade / Anomalia**: Tunelamento de DNS para Comando e Controle e Exfiltração de Dados (MITRE ATT&CK T1071.004 e T1048.003).\n\n"
            "**Impacto Potencial**: Fuga furtiva de dados confidenciais (DLP bypass) e manutenção de canal C2 estável para controle remoto de endpoints corporativos.\n\n"
            "</think>",
        "final_answer": "FATO: Os logs registram atividade inequívoca de DNS Tunneling (MITRE ATT&CK T1071.004 e T1048.003) para exfiltração de dados e comunicação de Comando e Controle (C2). O padrão é comprovado pelo volume anômalo de consultas do tipo `TXT` (> 1.200 queries em 10 minutos), subdomínios únicos de alta entropia contendo dados codificados e respostas volumosas sob o mesmo domínio apex `tunnelcorp.xyz`.\n\n"
            "DETECÇÃO / TELEMETRIA: Consulta KQL estruturada para Microsoft Sentinel / Azure Log Analytics para detecção de anomalia de entropia, FQDNs longos e volume excessivo de queries TXT:\n"
            "```kql\n"
            "DnsEvents\n"
            "| where TimeGenerated >= ago(1h)\n"
            "| where QueryType in (\"TXT\", \"NULL\")\n"
            "| extend DomainParts = split(Name, \".\")\n"
            "| extend ApexDomain = strcat(DomainParts[-2], \".\", DomainParts[-1])\n"
            "| extend Subdomain = tostring(DomainParts[0])\n"
            "| extend SubdomainLength = strlen(Subdomain)\n"
            "| where SubdomainLength > 25\n"
            "| summarize TotalQueries = count(), UniqueSubdomains = dcount(Name), AvgLength = avg(SubdomainLength) by ClientIP, ApexDomain\n"
            "| where TotalQueries > 100 and UniqueSubdomains > 50\n"
            "| order by TotalQueries desc\n"
            "```\n\n"
            "SUGESTÃO DE CORREÇÃO: 1) Configurar Response Policy Zones (RPZ) e sinkhole imediato no Bind/Unbound/CoreDNS corporativo para o domínio `tunnelcorp.xyz`; 2) Bloquear no firewall de borda a saída direta das portas UDP/TCP 53 e TCP 853 de qualquer máquina que não sejam os resolvedores DNS autorizados da empresa; 3) Habilitar inspeção de DNS com algoritmo de Machine Learning/Entropia no NGFW e bloquear tráfego DNS-over-HTTPS (DoH) não autorizado para provedores públicos."
    })

    # -------------------------------------------------------------------------
    # EXEMPLO 13: Autenticação Web / Criptografia (JWT Algorithm Confusion CWE-347)
    # -------------------------------------------------------------------------
    samples.append({
        "instruction": "Audite a rotina de verificação de autenticação JWT em Node.js apresentada abaixo. Avalie as vulnerabilidades relacionadas à manipulação do cabeçalho do token (`alg`), descreva o ataque de confusão de chave assimétrica para simétrica e forneça o código seguro com validação estrita.",
        "input": "const jwt = require('jsonwebtoken');\nconst fs = require('fs');\n\nconst publicKey = fs.readFileSync('./public.pem', 'utf8');\n\nfunction authenticateRequest(req, res, next) {\n    const authHeader = req.headers['authorization'];\n    if (!authHeader) {\n        return res.status(401).json({ error: \"Token ausente\" });\n    }\n\n    const token = authHeader.split(' ')[1];\n\n    try {\n        // Inseguro: a biblioteca confia cegamente no parametro 'alg' do header do token\n        const decoded = jwt.verify(token, publicKey);\n        req.user = decoded;\n        next();\n    } catch (err) {\n        return res.status(403).json({ error: \"Token invalido\" });\n    }\n}",
        "reasoning_trace": "<think>\n"
            "**Análise de Superfície**: O middleware Node.js `authenticateRequest` extrai um JSON Web Token (JWT) do cabeçalho HTTP `Authorization` e executa a verificação chamando `jwt.verify(token, publicKey)` utilizando uma chave pública RSA lida de um arquivo PEM.\n\n"
            "**Identificação de Pontos de Entrada / Vetores**: O ponto de entrada é o cabeçalho `Authorization: Bearer <token>`, controlado por qualquer cliente na internet.\n\n"
            "**Mapeamento de Fluxo / Mecânica Interna**: O token JWT consiste em `Header.Payload.Signature`. O cabeçalho declara o algoritmo de assinatura no campo `alg`. A chamada `jwt.verify(token, publicKey)` não especifica a lista restrita de algoritmos autorizados. Isso expõe a aplicação a duas falhas graves: (1) Confusão de Algoritmo de Chave Assimétrica para Simétrica (RS256 para HS256): A aplicação foi projetada para usar RSA assimétrica (chave pública para validar, chave privada para assinar). Se o invasor criar um token arbitrário (ex: `{\"user\": \"admin\", \"role\": \"superadmin\"}`), alterar o cabeçalho para `{\"alg\": \"HS256\"}` e assinar o token usando o algoritmo HMAC-SHA256 passando como chave secreta o conteúdo da própria chave pública PEM do servidor (`publicKey`), o servidor executará `jwt.verify` tratando a string da chave pública como o segredo compartilhado HMAC. Como a chave pública RSA é pública por definição, o invasor é capaz de assinar tokens arbitrários que serão validados com sucesso pelo backend; (2) Ataque de algoritmo nulo (`alg: none`): dependendo da versão da biblioteca, tokens sem assinatura são aceitos como válidos.\n\n"
            "**Detecção de Sinks Perigosos / Falhas de Controle**: O sink perigoso é a chamada `jwt.verify()` sem o parâmetro obrigatório de configuração `{ algorithms: ['RS256'] }`.\n\n"
            "**Avaliação de Sanitização e Mitigações**: Não há validação da integridade do algoritmo no cabeçalho e confia-se na declaração fornecida pelo próprio cliente no token.\n\n"
            "**Conclusão da Vulnerabilidade / Anomalia**: Vulnerabilidade de Verificação Inadequada de Assinatura Criptográfica / Confusão de Algoritmo (CWE-347 e CWE-327).\n\n"
            "**Impacto Potencial**: Bypass completo de autenticação e escalação de privilégios. Qualquer usuário anônimo pode emitir credenciais forjadas com privilégios de administrador do sistema.\n\n"
            "</think>",
        "final_answer": "FATO: O código é vulnerável ao ataque de Confusão de Algoritmo em JWT (CWE-347 / CWE-327). Como a chamada `jwt.verify(token, publicKey)` não define restrição estrita de algoritmos no parâmetro de opções, um atacante pode forjar um token alterando o cabeçalho de RS256 para HS256 e assiná-lo com o algoritmo simétrico HMAC-SHA256 utilizando como chave secreta a própria chave pública do servidor (`publicKey`). O servidor tratará a chave pública como o segredo HMAC, validando tokens arbitrários forjados pelo atacante.\n\n"
            "DETECÇÃO / TELEMETRIA: Alertas de WAF inspecionando tokens com `alg: none` ou incompatibilidade de assinatura. Regras Semgrep de análise estática (`javascript.jsonwebtoken.security.jwt-hardcode`) identificando chamadas de `jwt.verify` sem a opção `algorithms` especificada.\n\n"
            "SUGESTÃO DE CORREÇÃO: Configurar explicitamente a lista branca de algoritmos permitidos (`algorithms: ['RS256']`) e certificar-se de validar emissores (`issuer`) e audiências (`audience`):\n"
            "```javascript\n"
            "const jwt = require('jsonwebtoken');\n"
            "const fs = require('fs');\n\n"
            "const publicKey = fs.readFileSync('./public.pem', 'utf8');\n\n"
            "function authenticateRequestSecure(req, res, next) {\n"
            "    const authHeader = req.headers['authorization'];\n"
            "    if (!authHeader || !authHeader.startsWith('Bearer ')) {\n"
            "        return res.status(401).json({ error: \"Token de autenticacao ausente ou malformatado\" });\n"
            "    }\n\n"
            "    const token = authHeader.substring(7);\n\n"
            "    try {\n"
            "        // Forca estritamente apenas o algoritmo assimetrico esperado\n"
            "        const decoded = jwt.verify(token, publicKey, {\n"
            "            algorithms: ['RS256'],\n"
            "            issuer: 'https://auth.corp.internal'\n"
            "        });\n"
            "        req.user = decoded;\n"
            "        next();\n"
            "    } catch (err) {\n"
            "        return res.status(403).json({ error: \"Falha na validacao de assinatura do token\" });\n"
            "    }\n"
            "}\n"
            "```"
    })

    # -------------------------------------------------------------------------
    # EXEMPLO 14: Linux Kernel / Containers (Escape Docker via socket/CAP_SYS_ADMIN)
    # -------------------------------------------------------------------------
    samples.append({
        "instruction": "Analise o arquivo de composição Docker e as configurações de execução de container apresentadas a seguir. Identifique os vetores que permitem o escape do container para o host (Container Breakout), detalhe a mecânica do abuso do socket de gerenciamento e forneça a configuração corrigida e diretrizes de hardening.",
        "input": "version: '3.8'\n\nservices:\n  monitoring-agent:\n    image: internal-registry.corp/infra/monitor:v1.2\n    container_name: system_monitor\n    restart: always\n    privileged: true\n    cap_add:\n      - SYS_ADMIN\n    volumes:\n      - /var/run/docker.sock:/var/run/docker.sock\n      - /:/host:ro\n    ports:\n      - \"8080:8080\"",
        "reasoning_trace": "<think>\n"
            "**Análise de Superfície**: O arquivo `docker-compose.yml` define um serviço de monitoramento exposto na porta de rede 8080 configurado com privilégios máximos (`privileged: true`, `CAP_SYS_ADMIN`) e com o socket Unix do daemon Docker montado internamente (`/var/run/docker.sock`).\n\n"
            "**Identificação de Pontos de Entrada / Vetores**: O ponto de entrada inicial é o serviço de rede exposto na porta 8080. Qualquer vulnerabilidade de injeção de comandos, RCE ou falha na aplicação permite a obtenção de um shell dentro do container.\n\n"
            "**Mapeamento de Fluxo / Mecânica Interna**: A configuração oferece múltiplos caminhos diretos de escape do container para o host (Container Breakout - MITRE ATT&CK T1611): (1) Abuso do socket Docker (`/var/run/docker.sock`): O socket UNIX do daemon Docker fornece acesso irrestrito à API administrativa do Docker. Qualquer processo com permissão de escrita no socket pode instruir o daemon a executar um novo container no host com os seguintes parâmetros: montar o diretório raiz `/` do host com permissão de escrita e executar um comando com chroot ou criar uma chave SSH administrativa em `/root/.ssh/authorized_keys`; (2) Flag `privileged: true` e `CAP_SYS_ADMIN`: desativa todas as proteções padrão de Seccomp e AppArmor, expõe todos os dispositivos `/dev` do host e permite a manipulação de cgroups (abuso do gatilho `release_agent` para disparar scripts no host como root).\n\n"
            "**Detecção de Sinks Perigosos / Falhas de Controle**: O sink de controle é a montagem de um canal de controle irrestrito (`docker.sock`) associada à supressão completa dos mecanismos de isolamento do kernel Linux.\n\n"
            "**Avaliação de Sanitização e Mitigações**: O container não implementa o princípio do menor privilégio, executa como root e tem acesso direto ao plano de controle da infraestrutura.\n\n"
            "**Conclusão da Vulnerabilidade / Anomalia**: Execução com Privilégios Excessivos e Escape de Container (CWE-250 e MITRE ATT&CK T1611 - Escape to Host).\n\n"
            "**Impacto Potencial**: Comprometimento total da máquina hospedeira com privilégios de `root`, acesso a todos os outros containers residentes no mesmo nó, segredos de ambiente e controle total do cluster.\n\n"
            "</think>",
        "final_answer": "FATO: O container está configurado de forma insegura com `privileged: true`, capacidade `CAP_SYS_ADMIN` e montagem do socket do daemon Docker (`/var/run/docker.sock`). Essa combinação permite Escape de Container trivial (MITRE ATT&CK T1611): um processo comprometido dentro do container pode se comunicar diretamente com a API do Docker através do socket e criar um container privilegiado que monta o sistema de arquivos `/` do host com acesso total de root.\n\n"
            "DETECÇÃO / TELEMETRIA: Regras de runtime security Falco alertando interações suspeitas com o socket Docker a partir de containers:\n"
            "```yaml\n"
            "- rule: Acesso nao autorizado ao Docker Socket\n"
            "  desc: Detecta processos em containers acessando o docker socket\n"
            "  condition: open_write and fd.name = \"/var/run/docker.sock\" and container.id != host\n"
            "  output: Processo suspeito acessou o docker socket (user=%user.name command=%proc.cmdline container=%container.info)\n"
            "  priority: CRITICAL\n"
            "```\n\n"
            "SUGESTÃO DE CORREÇÃO: 1) Eliminar a montagem direta de `/var/run/docker.sock`. Se métricas forem necessárias, utilizar um proxy com restrição de verbos HTTP e endpoints (ex: Docker Socket Proxy permitindo apenas `GET /containers/json`); 2) Remover `privileged: true` e `cap_add: [SYS_ADMIN]`; 3) Executar o container como usuário sem privilégios (`user: 1000:1000`) com sistema de arquivos somente-leitura (`read_only: true`); 4) Aplicar perfis Seccomp e AppArmor restritos:\n"
            "```yaml\n"
            "version: '3.8'\n\n"
            "services:\n"
            "  monitoring-agent:\n"
            "    image: internal-registry.corp/infra/monitor:v1.2\n"
            "    container_name: system_monitor\n"
            "    restart: unless-stopped\n"
            "    read_only: true\n"
            "    user: \"10001:10001\"\n"
            "    security_opt:\n"
            "      - no-new-privileges:true\n"
            "      - apparmor:docker-default\n"
            "    cap_drop:\n"
            "      - ALL\n"
            "    ports:\n"
            "      - \"127.0.0.1:8080:8080\"\n"
            "```"
    })

    # -------------------------------------------------------------------------
    # EXEMPLO 15: Criptografia Aplicada (Reuso de Nonce em AES-GCM CWE-323)
    # -------------------------------------------------------------------------
    samples.append({
        "instruction": "Realize uma auditoria criptográfica na implementação em Python abaixo que utiliza AES-GCM para cifragem de comunicações confidenciais. Identifique a falha crítica no ciclo de vida dos vetores de inicialização (Nonce/IV), analise o impacto na integridade e confidencialidade matemática do modo Galois/Counter e forneça a implementação segura.",
        "input": "import os\nfrom cryptography.hazmat.primitives.ciphers.aead import AESGCM\n\nclass SecureMessagingService:\n    def __init__(self, key: bytes):\n        # Chave simétrica AES de 256 bits\n        self.key = key\n        self.cipher = AESGCM(self.key)\n        # Nonce fixo pre-definido para a sessao\n        self.session_nonce = b\"STATIC_NONCE_96\"\n\n    def encrypt_message(self, plaintext: bytes) -> bytes:\n        # Cifragem da mensagem utilizando o mesmo nonce fixo da sessao\n        ciphertext = self.cipher.encrypt(\n            self.session_nonce,\n            plaintext,\n            associated_data=None\n        )\n        return ciphertext",
        "reasoning_trace": "<think>\n"
            "**Análise de Superfície**: A classe Python `SecureMessagingService` utiliza a primitiva criptográfica AES-GCM de 256 bits (`AESGCM` da biblioteca `cryptography`). No construtor, define uma constante estática de 96 bits `self.session_nonce = b\"STATIC_NONCE_96\"` que é reutilizada em todas as chamadas do método `encrypt_message` sob a mesma chave simétrica.\n\n"
            "**Identificação de Pontos de Entrada / Vetores**: O ponto de entrada são as mensagens em texto claro fornecidas ao método `encrypt_message`. Um adversário com capacidade de captura de tráfego de rede (Man-in-the-Middle ou observador passivo) obtém múltiplos ciphertexts cifrados sob o mesmo par (Chave, Nonce).\n\n"
            "**Mapeamento de Fluxo / Mecânica Interna**: O modo AES-GCM combina a cifra de fluxo em modo CTR com um autenticador polinomial GHASH em campo de Galois $GF(2^{128})$: (1) Quebra de Confidencialidade: Em modo CTR, o texto cifrado é gerado via $C = P \\oplus \\text{Keystream}$. Ao reutilizar o mesmo Nonce com a mesma Chave, o Keystream é rigorosamente idêntico. O atacante calcula o XOR de dois ciphertexts: $C_1 \\oplus C_2 = (P_1 \\oplus K) \\oplus (P_2 \\oplus K) = P_1 \\oplus P_2$. Com técnicas de crib-dragging ou conhecimento parcial de cabeçalhos das mensagens, recupera-se ambos os textos claros integralmente; (2) Quebra Catastrófica de Autenticidade (The Forbidden Attack): A tag de autenticação em AES-GCM é calculada avaliando um polinômio com a subchave $H = E_K(0)$. A reutilização do Nonce permite deduzir a subchave de autenticação $H$ resolvendo uma equação polinomial sobre $GF(2^{128})$ a partir de duas tags. De posse de $H$, o atacante forja tags de autenticação válidas para mensagens arbitrárias sem conhecer a chave mestra AES.\n\n"
            "**Detecção de Sinks Perigosos / Falhas de Controle**: O sink é a chamada `self.cipher.encrypt(self.session_nonce, ...)` com um valor de Nonce invariável.\n\n"
            "**Avaliação de Sanitização e Mitigações**: Não há geração de vetor de inicialização fresco (random ou contador monotônico) por operação de cifragem.\n\n"
            "**Conclusão da Vulnerabilidade / Anomalia**: Reutilização de Nonce/IV em Cifra de Bloco Autenticada (CWE-323 - Reusing a Nonce, Key Pair in Encryption).\n\n"
            "**Impacto Potencial**: Perda completa de confidencialidade (recuperação de mensagens claras) e perda total de integridade/autenticidade (forjamento arbitrário de mensagens cifradas pelo atacante).\n\n"
            "</think>",
        "final_answer": "FATO: A classe `SecureMessagingService` é criticamente vulnerável à Reutilização de Nonce em AES-GCM (CWE-323). No modo GCM, a reutilização de um mesmo par (Chave, Nonce) acarreta o colapso criptográfico da cifra: permite a recuperação do texto claro via XOR de ciphertexts ($C_1 \\oplus C_2 = P_1 \\oplus P_2$) e possibilita a recuperação da subchave GHASH ('The Forbidden Attack'), permitindo que atacantes forjem tags de autenticação válidas e injetem dados adulterados sem detecção.\n\n"
            "DETECÇÃO / TELEMETRIA: Ferramentas de análise estática como Semgrep (`rules.crypto.aes-gcm-static-nonce`) identificando declarações de nonces estáticos. Em telemetria de aplicação, validação de unicidade de IVs em barramentos de mensageria com rejeição de IVs duplicados no cache de integridade.\n\n"
            "SUGESTÃO DE CORREÇÃO: Gerar um Nonce criptograficamente seguro de 96 bits (12 bytes) para cada mensagem individual via `os.urandom(12)`, transmitindo o Nonce em conjunto com o ciphertext:\n"
            "```python\n"
            "import os\n"
            "from cryptography.hazmat.primitives.ciphers.aead import AESGCM\n\n"
            "class SecureMessagingServiceFixed:\n"
            "    def __init__(self, key: bytes):\n"
            "        if len(key) not in (16, 32):\n"
            "            raise ValueError(\"A chave deve ter 128 ou 256 bits.\")\n"
            "        self.key = key\n"
            "        self.cipher = AESGCM(self.key)\n\n"
            "    def encrypt_message(self, plaintext: bytes) -> bytes:\n"
            "        # Geracao obrigatoria de Nonce de 96 bits unico e fresco por operacao\n"
            "        nonce = os.urandom(12)\n"
            "        ciphertext = self.cipher.encrypt(nonce, plaintext, associated_data=None)\n"
            "        # O nonce nao precisa ser secreto e eh transmitido prefixado ao ciphertext\n"
            "        return nonce + ciphertext\n\n"
            "    def decrypt_message(self, payload: bytes) -> bytes:\n"
            "        if len(payload) < 28: # 12 bytes nonce + 16 bytes auth tag minimo\n"
            "            raise ValueError(\"Carga util criptografada invalida ou corrompida.\")\n"
            "        nonce = payload[:12]\n"
            "        ciphertext = payload[12:]\n"
            "        return self.cipher.decrypt(nonce, ciphertext, associated_data=None)\n"
            "```"
    })

    # -------------------------------------------------------------------------
    # EXEMPLO 16: PHP / Web Security (Path Traversal + LFI CWE-22 / CWE-98)
    # -------------------------------------------------------------------------
    samples.append({
        "instruction": "Analise o script PHP de renderização de páginas abaixo. Identifique vulnerabilidades de manipulação de caminhos de arquivos (Path Traversal / LFI), avalie a ineficácia da substituição de caracteres (bypass de sanitização) e apresente a arquitetura de resolução segura de arquivos.",
        "input": "<?php\n// Endpoint de carregamento de modulos da aplicacao\nif (isset($_GET['page'])) {\n    $page = $_GET['page'];\n\n    // Sanitizacao ingenua: tenta remover sequencias de retrocesso de diretorio\n    $sanitized_page = str_replace(\"../\", \"\", $page);\n\n    // Carregamento dinamico de arquivos locais\n    $file_path = \"templates/\" . $sanitized_page . \".php\";\n\n    if (file_exists($file_path)) {\n        include($file_path);\n    } else {\n        echo \"Pagina nao encontrada.\";\n    }\n}\n?>",
        "reasoning_trace": "<think>\n"
            "**Análise de Superfície**: O script PHP captura a variável de requisição `$_GET['page']`, tenta remover a substring `../` utilizando `str_replace`, concatena o resultado no caminho base `\"templates/\" . $sanitized_page . \".php\"` e executa a inclusão dinâmica do arquivo no servidor através da instrução `include($file_path)`.\n\n"
            "**Identificação de Pontos de Entrada / Vetores**: O ponto de entrada é o parâmetro de URL `$_GET['page']`, enviado por qualquer cliente via requisições HTTP GET normais.\n\n"
            "**Mapeamento de Fluxo / Mecânica Interna**: A tentativa de proteção com `str_replace(\"../\", \"\", $page)` é fundamentalmente ineficaz por não ser recursiva: (1) Bypass de substituição: Se o invasor enviar `....//` (ou `..././`), o `str_replace` remove apenas a ocorrência central `../`, recompondo os caracteres restantes em uma nova sequência válida de retrocesso `../`; (2) Inclusão Local de Arquivos (LFI): O invasor fornece `....//....//....//etc/passwd` ou aponta para arquivos de configuração internos da aplicação contendo senhas e tokens de API; (3) Escalação para RCE: Se o ambiente contiver funcionalidades de upload de avatar, logs de acesso acessíveis (Apache access.log / PHP session files) ou suporte a wrappers (`php://input`, `data://`), o invasor injeta código PHP (`<?php system($_GET['cmd']); ?>`) nesses arquivos e força a inclusão via LFI, resultando em Execução Remota de Código.\n\n"
            "**Detecção de Sinks Perigosos / Falhas de Controle**: O sink perigoso é a diretiva `include($file_path)`, que não apenas lê o arquivo, mas executa qualquer bloco de código PHP presente no conteúdo.\n\n"
            "**Avaliação de Sanitização e Mitigações**: A sanitização baseada em exclusão superficial de strings não impede sequências recursivas, não valida o caminho absoluto com `realpath()` e não confina a busca a um conjunto restrito de templates autorizados.\n\n"
            "**Conclusão da Vulnerabilidade / Anomalia**: Vulnerabilidade de Path Traversal (CWE-22) combinada com Local File Inclusion (LFI - CWE-98).\n\n"
            "**Impacto Potencial**: Divulgação de dados confidenciais do servidor (leitura de arquivos do sistema operacional e código-fonte) e Execução Remota de Código (RCE) via envenenamento de logs (Log Poisoning) ou inclusão de arquivos temporários.\n\n"
            "</think>",
        "final_answer": "FATO: O script PHP é vulnerável a Path Traversal (CWE-22) e Local File Inclusion (LFI - CWE-98). A função `str_replace(\"../\", \"\", $page)` não é recursiva e pode ser contornada por sequências como `....//` ou URL encoding (`%2e%2e%2f`). O caminho resultante é passado diretamente para o comando `include()`, permitindo que um atacante leia arquivos arbitrários do sistema e escale para Execução Remota de Código (RCE) caso consiga envenenar logs de servidor ou arquivos de sessão.\n\n"
            "DETECÇÃO / TELEMETRIA: Regras de WAF do OWASP ModSecurity Core Rule Set (regras 930100 e 930110: \"Path Traversal Attack\"). Logs de acesso web (Nginx/Apache) monitorando requisições com código 200 contendo padrões `..%2f`, `/etc/passwd` ou esquemas de wrapper `php://`.\n\n"
            "SUGESTÃO DE CORREÇÃO: 1) Utilizar uma lista branca estrita (allowlist) de templates autorizados (abordagem mais segura); 2) Caso arquivos dinâmicos sejam necessários, extrair o nome canônico do arquivo com `basename()` e validar com `realpath()` assegurando que o arquivo resolvido pertença exclusivamente ao diretório base autorizado:\n"
            "```php\n"
            "<?php\n"
            "if (isset($_GET['page'])) {\n"
            "    $page = $_GET['page'];\n\n"
            "    // Abordagem segura 1: Whitelist estrita de templates autorizados\n"
            "    $allowed_pages = [\n"
            "        'home'     => 'templates/home.php',\n"
            "        'about'    => 'templates/about.php',\n"
            "        'contact'  => 'templates/contact.php'\n"
            "    ];\n\n"
            "    if (array_key_exists($page, $allowed_pages)) {\n"
            "        include($allowed_pages[$page]);\n"
            "        exit;\n"
            "    }\n\n"
            "    // Abordagem segura 2 (defesa em profundidade com realpath)\n"
            "    $base_dir = realpath(__DIR__ . '/templates');\n"
            "    $sanitized_filename = basename($page) . '.php';\n"
            "    $target_file = realpath($base_dir . '/' . $sanitized_filename);\n\n"
            "    if ($target_file && strpos($target_file, $base_dir) === 0 && file_exists($target_file)) {\n"
            "        include($target_file);\n"
            "    } else {\n"
            "        http_response_code(404);\n"
            "        echo \"Pagina solicitada nao encontrada.\";\n"
            "    }\n"
            "}\n"
            "?>\n"
            "```"
    })

    return samples

def main():
    samples = get_all_samples()
    print(f"[*] Construindo dataset com {len(samples)} amostras de alta densidade analitica...")

    # Garante criacao do diretorio pai se necessario
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples, start=1):
            line = json.dumps(sample, ensure_ascii=False)
            f.write(line + "\n")
            print(f"  [+] Amostra {idx} gravada com sucesso: {sample['instruction'][:60]}...")

    print("=" * 60)
    print(f"[OK] Total de {len(samples)} amostras gravadas com sucesso em:")
    print(f"     {OUTPUT_JSONL}")
    print("=" * 60)

if __name__ == "__main__":
    main()
