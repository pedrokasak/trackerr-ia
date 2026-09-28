"""
Guardrail minimo do resumo de documento de RI (TRA-238).

Por que nao reusar rag/response_guard.py: aquele barra qualquer
"venda"/"investir"/"recomendacao", o que e certo pra resposta sobre a
carteira do usuario, mas texto de RI usa essas palavras como FATO — "venda
da subsidiaria", "a companhia vai investir R$ 2 bi", "recomendacao do
conselho de pagar JCP". Reusar aquele guard rejeitaria quase todo release.

Duas camadas, com consequencias diferentes:

- `validate_ri_summary`: recomendacao DIRIGIDA ao leitor (imperativo ou
  segunda pessoa) e preco-alvo proprio. Rejeita o resumo INTEIRO — se o
  modelo desobedeceu a regra central, o resto dele tambem nao merece
  confianca.
- `enforce_fidelity` (TRA-239): cada destaque precisa de um trecho literal
  do documento que o sustente, e todo numero do destaque precisa estar
  nesse trecho. Destaque sem suporte e DESCARTADO, os demais seguem — um
  numero arredondado nao invalida os outros sete destaques corretos.

Mesmo principio do guard do RAG: a garantia fica em codigo, nao em
instrucao de prompt.
"""

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from ri.fidelity import (
    SourceIndex,
    excerpt_span,
    extract_identifiers,
    number_set,
    page_at,
    strip_page_markers,
    unsupported_identifiers,
    unsupported_numbers,
)

# Trecho exibido como citacao: o suficiente pra conferir, sem virar copia
# do documento.
MAX_EXCERPT_CHARS = 300

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


@dataclass(frozen=True)
class RawHighlight:
    """Destaque como o modelo devolveu: afirmacao + trecho que a sustenta."""

    text: str
    evidence: str


@dataclass(frozen=True)
class CitedHighlight:
    """Destaque aprovado, com o trecho REAL do documento e a pagina dele."""

    text: str
    excerpt: str
    page: Optional[int]


@dataclass
class FidelityResult:
    highlights: List[CitedHighlight] = field(default_factory=list)
    narrative: str = ""
    # Motivo de cada descarte, para log e metrica — nunca o conteudo.
    dropped: List[str] = field(default_factory=list)


def enforce_fidelity(
    highlights: Sequence[RawHighlight],
    narrative: str,
    source: SourceIndex,
    metadata: Iterable[str],
) -> FidelityResult:
    """
    Mantem so o que o documento sustenta.

    `metadata` (ticker, empresa, periodo) libera IDENTIFICADORES: o release
    do 2T26 da PETR4 pode nao escrever "PETR4" nem "2T26" no corpo. Numero
    de metadado nao e liberado — os dias e meses da data de publicacao
    abririam passagem para qualquer "8%" inventado.
    """
    allowed_identifiers = source.identifiers | frozenset(
        extract_identifiers(" ".join(item for item in metadata if item))
    )
    result = FidelityResult()

    for highlight in highlights:
        span = excerpt_span(source, highlight.evidence)
        if span is None:
            result.dropped.append("highlight_evidence_not_found")
            continue
        excerpt = source.text[span[0] : span[1]]
        if unsupported_numbers(highlight.text, number_set(excerpt)):
            result.dropped.append("highlight_number_not_in_evidence")
            continue
        if unsupported_identifiers(highlight.text, allowed_identifiers, source):
            result.dropped.append("highlight_unknown_identifier")
            continue
        result.highlights.append(
            CitedHighlight(
                text=highlight.text,
                excerpt=strip_page_markers(excerpt)[:MAX_EXCERPT_CHARS],
                page=page_at(source, span[0]),
            )
        )

    narrative = str(narrative or "").strip()
    if narrative and (
        unsupported_numbers(narrative, source.numbers)
        or unsupported_identifiers(narrative, allowed_identifiers, source)
    ):
        # A narrativa nao tem trecho proprio: o teste e contra o documento
        # inteiro. Sem suporte, sai inteira — cortar frases do meio deixaria
        # "Alem disso..." apontando pra nada.
        result.dropped.append("narrative_unsupported_claim")
        narrative = ""
    result.narrative = narrative
    return result
