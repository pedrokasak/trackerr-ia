"""
Guardrail da resposta do RAG (TRA-37): nunca recomendação de compra/venda,
nunca número definitivo de imposto. Mesmo princípio do digest de e-mail
(server: digest-narrative-validator.ts) — a garantia fica em código, não
em instrução de prompt. O disclaimer NÃO é validado aqui: é anexado
sempre, deterministicamente, pelo chamador (rag/query_service.py), então
não existe caminho onde ele fica ausente por o modelo ter esquecido.
"""

import re

RECOMMENDATION_PATTERN = re.compile(
    r"\b(compre|comprar|venda|vender|recomendo|recomendamos|recomendação|"
    r"invista|investir)\b",
    re.IGNORECASE,
)

# Frases que afirmam um numero de imposto como fato fechado. O motor fiscal
# deterministico (TRA-40) ainda nao existe — ate existir, nenhuma resposta
# do RAG pode soar como calculo definitivo, so como estimativa educativa.
DEFINITIVE_TAX_CLAIM_PATTERN = re.compile(
    r"\b(você deve pagar|o valor devido é|está isento de pagar|"
    r"o imposto devido é)\b",
    re.IGNORECASE,
)


class ResponseGuardResult:
    def __init__(self, valid: bool, reason: str | None = None) -> None:
        self.valid = valid
        self.reason = reason

    def __repr__(self) -> str:
        return f"ResponseGuardResult(valid={self.valid}, reason={self.reason!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResponseGuardResult):
            return NotImplemented
        return self.valid == other.valid and self.reason == other.reason


def validate_rag_response(text: str) -> ResponseGuardResult:
    trimmed = (text or "").strip()
    if not trimmed:
        return ResponseGuardResult(False, "empty")
    if RECOMMENDATION_PATTERN.search(trimmed):
        return ResponseGuardResult(False, "recommendation_language")
    if DEFINITIVE_TAX_CLAIM_PATTERN.search(trimmed):
        return ResponseGuardResult(False, "definitive_tax_claim")
    return ResponseGuardResult(True)
