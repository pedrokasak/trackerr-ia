"""
Guardrail minimo do resumo de documento de RI (TRA-238).

Por que nao reusar rag/response_guard.py: aquele barra qualquer
"venda"/"investir"/"recomendacao", o que e certo pra resposta sobre a
carteira do usuario, mas texto de RI usa essas palavras como FATO — "venda
da subsidiaria", "a companhia vai investir R$ 2 bi", "recomendacao do
conselho de pagar JCP". Reusar aquele guard rejeitaria quase todo release.

Aqui so e barrado o que nunca pode sair num resumo de documento publico:
recomendacao DIRIGIDA ao leitor (imperativo ou segunda pessoa) e preco-alvo
proprio. Checagem de fidelidade numerica e citacao por pagina ficam em
TRA-239. Mesmo principio do guard do RAG: a garantia fica em codigo, nao em
instrucao de prompt.
"""

import re
from dataclasses import dataclass
from typing import Iterable, Optional

# Imperativo no inicio de frase ("Compre", "Venda suas acoes") ou fala
# dirigida ao leitor ("voce deveria vender", "recomendamos a compra").
# "Venda" solta NAO entra: "Venda da subsidiaria concluida" e fato.
DIRECTED_RECOMMENDATION_PATTERN = re.compile(
    r"(?:^|[.!?]\s*)(?:compre|comprem|venda\s+(?:suas|seus|a\s+a[çc][ãa]o|o\s+papel)|vendam|invista|invistam)\b"
    r"|\bvoc[êe]s?\s+(?:deve(?:ria)?m?|precisa(?:m)?|pode(?:m)?)\s+(?:comprar|vender|investir|aumentar|reduzir|zerar)\b"
    r"|\brecomend(?:amos|o)\s+(?:a\s+)?(?:compra|venda|que\s+voc[êe])\b"
    r"|\b(?:oportunidade\s+de\s+compra|hora\s+de\s+(?:comprar|vender))\b",
    re.IGNORECASE | re.MULTILINE,
)

PRICE_TARGET_PATTERN = re.compile(
    r"\bpre[çc]o[-\s]?alvo\b|\btarget\s+price\b|\bpotencial\s+de\s+valoriza[çc][ãa]o\s+de\s+\d",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RiSummaryGuardResult:
    valid: bool
    reason: Optional[str] = None


def validate_ri_summary(
    highlights: Iterable[str], narrative: str
) -> RiSummaryGuardResult:
    parts = [str(item).strip() for item in highlights or [] if str(item).strip()]
    narrative_text = str(narrative or "").strip()
    if narrative_text:
        parts.append(narrative_text)

    if not parts:
        return RiSummaryGuardResult(False, "empty")

    # Cada destaque e checado isolado: o "^" do imperativo precisa casar com
    # o inicio de CADA frase, e juntar tudo numa string so esconderia isso.
    for text in parts:
        if DIRECTED_RECOMMENDATION_PATTERN.search(text):
            return RiSummaryGuardResult(False, "directed_recommendation")
        if PRICE_TARGET_PATTERN.search(text):
            return RiSummaryGuardResult(False, "price_target")

    return RiSummaryGuardResult(True)
