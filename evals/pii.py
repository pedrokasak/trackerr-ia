"""
Minimização antes do juiz (TRA-242, LGPD): o que sai para um provider de
LLM passa por aqui. Documentos e contatos viram marcadores; o `user_id`
nunca entra no item avaliado.

Detecção por padrão, com limites conhecidos: nome próprio solto na pergunta
não é detectável com segurança e não é removido. Por isso a amostra é só de
pergunta e resposta, sem o perfil do usuário, e o relatório final não guarda
texto nenhum.
"""

import re
from typing import List, Tuple

# Ordem importa: CNPJ antes de CPF (o CPF casaria dentro do CNPJ), cartão
# antes de telefone.
_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("[EMAIL]", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("[CNPJ]", re.compile(r"(?<!\d)\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}(?!\d)")),
    ("[CPF]", re.compile(r"(?<!\d)\d{3}\.\d{3}\.\d{3}-\d{2}(?!\d)|(?<![\d.,])\d{11}(?![\d.,])")),
    ("[CARTAO]", re.compile(r"(?<!\d)(?:\d{4}[ -]){3}\d{1,7}(?!\d)")),
    (
        "[TELEFONE]",
        re.compile(
            r"(?:\+?55\s?)?\(\d{2}\)\s?9?\d{4}-?\d{4}(?!\d)"
            r"|(?<!\d)\+55\s?\d{2}\s?9?\d{4}-?\d{4}(?!\d)"
            r"|(?<![\d.,])\d{2}\s9\d{4}-\d{4}(?!\d)"
        ),
    ),
    (
        "[CONTA]",
        re.compile(r"\b(ag[eê]ncia|conta(?:\s+corrente)?)\s*:?\s*(?:n[ºo°]\s*)?\d[\d.-]*", re.IGNORECASE),
    ),
]


def scrub(text: str) -> str:
    """Texto com documentos, contatos e contas trocados por marcadores."""
    cleaned = str(text or "")
    for placeholder, pattern in _PATTERNS:
        cleaned = pattern.sub(placeholder, cleaned)
    return cleaned


def contains_pii(text: str) -> bool:
    """Ainda há algo com cara de dado pessoal? Usado como trava final."""
    value = str(text or "")
    return any(pattern.search(value) for _, pattern in _PATTERNS)
