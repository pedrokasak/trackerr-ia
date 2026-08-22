"""
Frescor do dado no caminho de resposta do RAG (TRA-77).

A ingestao roda em ciclo (cron diario, TRA-84). Entre um ciclo e outro — e
principalmente quando um ciclo falha — os chunks envelhecem. Sem checar
idade, o RAG recupera um chunk de 5 dias atras e o LLM narra com a mesma
confianca de um dado de hoje. Pra dinheiro e carteira, dado velho
apresentado como atual e pior que ausencia de resposta.

A degradacao acontece em CODIGO, nao em instrucao de prompt — mesmo
principio ja usado pro disclaimer e pro deny-list de recomendacao. Frescor
e garantia, nao sugestao que se confia ao modelo.
"""

from dataclasses import dataclass
from datetime import date

# Limite de frescor por source_type, em dias. Posicao de carteira envelhece
# mais rapido que perfil de risco: o preco muda todo dia, o perfil de risco
# estrutural nao. Acima do limite, a resposta e anotada explicitamente.
FRESHNESS_THRESHOLD_DAYS: dict[str, int] = {
    "portfolio_position": 2,
    "portfolio_performance": 3,
    "portfolio_dividend": 7,
    "portfolio_risk": 7,
}
DEFAULT_THRESHOLD_DAYS = 3

# Acima disto, nao basta anotar — o dado esta velho demais pra narrar como se
# fosse retrato atual. A resposta leva um aviso forte no topo.
HARD_STALE_MULTIPLIER = 3


@dataclass
class FreshnessAssessment:
    # Maior idade (em dias) entre os chunks recuperados, relativa ao seu
    # proprio limite. None quando nenhum chunk tem as_of conhecido.
    max_age_days: int | None
    is_stale: bool
    is_hard_stale: bool
    # Nota pra prefixar na resposta quando stale. Vazia quando fresco.
    note: str


def _age_days(as_of: date | None, today: date) -> int | None:
    if as_of is None:
        return None
    return (today - as_of).days


def assess_freshness(chunks: list, today: date) -> FreshnessAssessment:
    """
    Avalia o frescor do pior chunk relevante. Um unico chunk velho ja
    contamina a resposta, entao a avaliacao e pelo MAIOR excesso de idade
    sobre o limite do proprio source_type, nao pela media.
    """
    worst_excess = 0  # dias acima do limite
    worst_age: int | None = None
    any_known = False

    for chunk in chunks:
        age = _age_days(getattr(chunk, "as_of", None), today)
        if age is None:
            continue
        any_known = True
        threshold = FRESHNESS_THRESHOLD_DAYS.get(
            getattr(chunk, "source_type", ""), DEFAULT_THRESHOLD_DAYS
        )
        excess = age - threshold
        if excess > worst_excess:
            worst_excess = excess
            worst_age = age
        elif worst_age is None or age > worst_age:
            worst_age = max(worst_age or 0, age)

    if not any_known:
        # Nenhum chunk tem data — nao da pra afirmar frescor. Conservador,
        # mas sem bloquear: anota que a data nao pode ser confirmada.
        return FreshnessAssessment(
            max_age_days=None,
            is_stale=False,
            is_hard_stale=False,
            note="",
        )

    if worst_excess <= 0:
        return FreshnessAssessment(
            max_age_days=worst_age, is_stale=False, is_hard_stale=False, note=""
        )

    is_hard = worst_excess > (DEFAULT_THRESHOLD_DAYS * HARD_STALE_MULTIPLIER)
    days = worst_age or 0
    if is_hard:
        note = (
            f"⚠️ Atenção: os dados da sua carteira usados nesta resposta têm "
            f"cerca de {days} dias e podem estar desatualizados. Confirme a "
            f"posição atual antes de qualquer decisão."
        )
    else:
        note = (
            f"Observação: esta resposta usa dados da sua carteira de "
            f"aproximadamente {days} dias atrás."
        )
    return FreshnessAssessment(
        max_age_days=days, is_stale=True, is_hard_stale=is_hard, note=note
    )
