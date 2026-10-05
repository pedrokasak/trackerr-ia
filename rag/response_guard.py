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

# Recusa explicita do proprio modelo: "Nao posso fazer recomendacoes de
# compra, venda ou qualquer acao". Ela cita as palavras justamente para
# negar a recomendacao — e o guardrail barrava a resposta que obedecia a
# regra, trocando os fatos pelo "nao consigo responder" (achado da avaliacao
# offline, TRA-242). So a recusa sai da varredura, e so quando o resto da
# frase e vocabulario da propria recusa (ver `_REFUSAL_VOCABULARY`):
# "nao venda agora" e ordem, nao recusa, e continua barrado; "nao posso
# fazer recomendacoes, compre PETR4" tambem.
REFUSAL_PATTERN = re.compile(
    r"n[aã]o\s+(?:posso|consigo|devo|vou|fa[cç]o|[eé]\s+poss[ií]vel)\s+"
    r"(?:fazer\s+|dar\s+|oferecer\s+)?"
    r"(?:recomenda[cç](?:[aã]o|[oõ]es)|indica[cç](?:[aã]o|[oõ]es)|sugest(?:[aã]o|[oõ]es))"
    r"([^.!?\n;]*)",
    re.IGNORECASE,
)

# Palavras que podem completar a recusa ("...de compra, venda ou qualquer
# acao sobre seus ativos"). Qualquer outra — um ticker, "compre", "se
# quiser", "mas" — e conteudo, e a frase inteira volta para a varredura.
_REFUSAL_VOCABULARY = frozenset(
    """
    de da do das dos e ou a o as os um uma em no na nos nas para sobre com
    qualquer quaisquer relacao relação seu sua seus suas voce você
    compra compras venda vendas comprar vender manter manutencao manutenção
    acao ação acoes ações operacao operação operacoes operações
    ativo ativos papel papeis papéis titulo título titulos títulos
    investimento investimentos aplicacao aplicação aplicacoes aplicações
    especifica específica especificas específicas especifico específico
    personalizada personalizadas individual individuais financeira financeiras
    financeiro financeiros carteira mercado
    """.split()
)

_WORD = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def _strip_refusals(text: str) -> str:
    """Tira da varredura a recusa explicita, e so ela."""

    def replace(match: re.Match) -> str:
        tail_words = _WORD.findall(match.group(1).lower())
        if all(word in _REFUSAL_VOCABULARY for word in tail_words):
            return " "
        return match.group(0)

    return REFUSAL_PATTERN.sub(replace, text)


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
    if RECOMMENDATION_PATTERN.search(_strip_refusals(trimmed)):
        return ResponseGuardResult(False, "recommendation_language")
    if DEFINITIVE_TAX_CLAIM_PATTERN.search(trimmed):
        return ResponseGuardResult(False, "definitive_tax_claim")
    return ResponseGuardResult(True)
