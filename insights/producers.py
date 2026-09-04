"""
Producers legados migrados para o shape estendido de Insight (TRA-135).

Antes cada um vinha como um bloco de JSON livre dentro do prompt
monolitico de `/api/hybrid-analysis` (opportunity_radar, error_detection,
smart_feed, recommendations). LLM inventava numero e categoria a vontade —
mesmo problema que motivou TRA-55 e TRA-56.

Aqui cada producer devolve `List[InsightCandidate]` com evidencia
deterministica calculada a partir da carteira; a narracao ("rationale" e
"action_label") passa pelo mesmo pipeline de `InsightsService` — LLM so
escreve prosa a partir da evidencia, parser rejeita numero fora da
evidencia, fallback deterministico quando o LLM falha.

IDs sao namespaced por producer (ex.: `opportunity_radar.mover.BTC`) pra
nao colidir com `build_deterministic_evidence` (concentration.* /
rebalance.*) — a mesma carteira pode aparecer em varios producers, cada um
com seu foco.
"""

from __future__ import annotations

from typing import List, Optional

from models.models import UserProfile

from insights.service import (
    CATEGORY_DEVIATION_PP,
    IDEAL_ALLOCATION,
    InsightCandidate,
    SINGLE_ASSET_CONCENTRATION_LIMIT_PCT,
    _normalize_category,
    exposure_by_category,
    top_asset_concentration,
)
from models.models import InsightAction, InsightEvidence


# ---------------------------------------------------------------------------
# Thresholds compartilhados
# ---------------------------------------------------------------------------

# |change_24h| a partir do qual o ativo entra no smart_feed / opportunity.
# 3% e a faixa em que uma variacao 24h ja e conversavel pro usuario final
# sem virar ruido diario.
NOTABLE_MOVE_PCT = 3.0

# Alcance do feed: pra evitar payload gigante em carteiras grandes.
FEED_MAX_ITEMS = 5

# FII com pvp acima disso e "critical_rejection" na FiiStrategy — puxamos o
# mesmo limiar aqui pra manter uma unica fonte de verdade.
FII_PVP_ALERT = 1.5

# Desvio maior que este pp por categoria vira alerta critico em
# error_detection (mais estrito que o CATEGORY_DEVIATION_PP usado no
# rebalanceador padrao — aqui e "erro", nao "ajuste").
ERROR_CATEGORY_DEVIATION_PP = 20.0


def _asset_weight_pct(profile: UserProfile, symbol: str) -> Optional[float]:
    total = profile.portfolio.total_value or sum(
        (a.quantity or 0) * (a.current_price or 0) for a in profile.portfolio.assets
    )
    if total <= 0:
        return None
    for asset in profile.portfolio.assets:
        if asset.symbol == symbol:
            value = (asset.quantity or 0) * (asset.current_price or 0)
            return round(value / total * 100, 2)
    return None


# ---------------------------------------------------------------------------
# smart_feed
# ---------------------------------------------------------------------------


def build_smart_feed_evidence(profile: UserProfile) -> List[InsightCandidate]:
    """
    TRA-135: smart_feed migrado para o shape estendido.

    Antes o LLM inventava titulos como "Bitcoin puxou +1.2%" a partir de
    nada. Agora: ordena ativos por |change_24h|, corta em
    NOTABLE_MOVE_PCT, cita valor + peso na carteira. Sem rota — feed e
    observacao, nao acao (action=None e valido no shape).
    """

    movers = sorted(
        (a for a in profile.portfolio.assets if (a.change_24h or 0) != 0),
        key=lambda a: abs(a.change_24h or 0),
        reverse=True,
    )
    candidates: List[InsightCandidate] = []
    for asset in movers[:FEED_MAX_ITEMS]:
        change = asset.change_24h or 0
        if abs(change) < NOTABLE_MOVE_PCT:
            break
        weight = _asset_weight_pct(profile, asset.symbol)
        direction = "alta" if change > 0 else "queda"
        evidence = [
            InsightEvidence(
                label=f"Variacao 24h de {asset.symbol}",
                value=round(change, 2),
                source=f"portfolio.asset.{asset.symbol}.change_24h",
            ),
        ]
        if weight is not None:
            evidence.append(
                InsightEvidence(
                    label=f"Peso de {asset.symbol} na carteira",
                    value=weight,
                    source=f"portfolio.asset.{asset.symbol}.weight",
                )
            )
        weight_txt = (
            f" (peso {weight}% da carteira)" if weight is not None else ""
        )
        candidates.append(
            InsightCandidate(
                id=f"smart_feed.mover.{asset.symbol}",
                title=f"{asset.symbol} em {direction} 24h",
                body=f"{asset.symbol} variou {change:+.2f}% nas ultimas 24h{weight_txt}.",
                evidence=evidence,
                action=None,
                fallback_rationale=(
                    f"{asset.symbol} teve variacao de {change:+.2f}% em 24h"
                    f"{weight_txt}. Movimento fica registrado no feed pra "
                    "contexto, sem sugerir ordem de compra ou venda."
                ),
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# opportunity_radar
# ---------------------------------------------------------------------------


def build_opportunity_radar_evidence(profile: UserProfile) -> List[InsightCandidate]:
    """
    TRA-135: opportunity_radar migrado.

    Legado (o campo `opportunity_radar` do prompt monolitico) pedia ao LLM
    3 "observacoes sobre o que mudou". Sem dado real de watchlist externa
    o LLM inventava ticker e preco (motivo do TRA-55). Aqui restringimos
    o radar ao que a carteira ja carrega: ativos com queda relevante em 24h
    e peso material — vale a pena OBSERVAR se a tese ainda vale. Sem
    action.route pra nao virar sugestao de operacao.

    TODO(TRA-135): quando o server passar a mandar watchlist externa
    (noticia setorial, evento macro) o radar deve puxar dai. Por ora usa
    so o `change_24h` que ja vem no UserProfile.
    """

    total = profile.portfolio.total_value or sum(
        (a.quantity or 0) * (a.current_price or 0) for a in profile.portfolio.assets
    )
    if total <= 0:
        return []
    candidates: List[InsightCandidate] = []
    for asset in profile.portfolio.assets:
        change = asset.change_24h or 0
        if change >= -NOTABLE_MOVE_PCT:
            # Radar de observacao pos-queda: sobe nao entra pra nao virar
            # gatilho de compra escondido.
            continue
        weight = _asset_weight_pct(profile, asset.symbol) or 0.0
        if weight < 5.0:
            # Ativo minusculo cair 3% nao pauta atencao — evita ruido.
            continue
        candidates.append(
            InsightCandidate(
                id=f"opportunity_radar.watch.{asset.symbol}",
                title=f"Vale observar {asset.symbol}",
                body=(
                    f"{asset.symbol} caiu {change:+.2f}% em 24h e representa "
                    f"{weight}% da carteira."
                ),
                evidence=[
                    InsightEvidence(
                        label=f"Variacao 24h de {asset.symbol}",
                        value=round(change, 2),
                        source=f"portfolio.asset.{asset.symbol}.change_24h",
                    ),
                    InsightEvidence(
                        label=f"Peso de {asset.symbol} na carteira",
                        value=weight,
                        source=f"portfolio.asset.{asset.symbol}.weight",
                    ),
                    InsightEvidence(
                        label="Limiar de movimento notavel (%)",
                        value=NOTABLE_MOVE_PCT,
                        source="policy.notable_move_pct",
                    ),
                ],
                action=None,
                fallback_rationale=(
                    f"{asset.symbol} teve variacao de {change:+.2f}% em 24h e "
                    f"pesa {weight}% da carteira, acima do limiar de "
                    f"{NOTABLE_MOVE_PCT}%. Vale revisar a tese; o radar so "
                    "descreve o movimento, nao emite ordem."
                ),
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# error_detection
# ---------------------------------------------------------------------------


def build_error_detection_evidence(profile: UserProfile) -> List[InsightCandidate]:
    """
    TRA-135: error_detection migrado.

    Reune erros CRITICOS a partir de regras explicitas — nao "sensacao" do
    LLM. Cobre: concentracao acima do limite de ativo unico, desvio de
    categoria acima do limiar critico (20pp), FII com pvp > 1.5 (mesmo
    limiar do `critical_rejection` em FiiStrategy).

    IDs prefixados com `error_detection.` pra nao colidir com os insights
    padrao (concentration.* / rebalance.*), que continuam sendo emitidos
    pelo `/api/insights` com foco em ajuste, nao em erro.
    """

    candidates: List[InsightCandidate] = []
    top = top_asset_concentration(profile)
    if top and top["pct"] >= SINGLE_ASSET_CONCENTRATION_LIMIT_PCT:
        symbol = top["symbol"]
        pct = top["pct"]
        candidates.append(
            InsightCandidate(
                id=f"error_detection.concentration.{symbol}",
                title=f"Concentracao critica em {symbol}",
                body=f"{symbol} representa {pct}% da carteira.",
                evidence=[
                    InsightEvidence(
                        label=f"Peso de {symbol} na carteira",
                        value=pct,
                        source=f"portfolio.asset.{symbol}",
                    ),
                    InsightEvidence(
                        label="Limite de concentracao por ativo",
                        value=SINGLE_ASSET_CONCENTRATION_LIMIT_PCT,
                        source="policy.single_asset_concentration_limit",
                    ),
                ],
                action=InsightAction(
                    label=f"Abrir rebalanceador para reduzir {symbol}",
                    route="/rebalancer",
                    payload={
                        "symbol": symbol,
                        "target_max_pct": SINGLE_ASSET_CONCENTRATION_LIMIT_PCT,
                    },
                    why=(
                        f"{symbol} passou do limite de concentracao por ativo "
                        "unico — risco idiossincratico alto."
                    ),
                ),
                fallback_rationale=(
                    f"{symbol} responde por {pct}% da carteira, acima do "
                    f"limite de {SINGLE_ASSET_CONCENTRATION_LIMIT_PCT}%. "
                    "Concentracao dessa ordem transforma o risco especifico "
                    "do ativo em risco material do patrimonio."
                ),
            )
        )

    exposure = exposure_by_category(profile)
    risk = (profile.risk_profile or "moderate").lower()
    ideal = IDEAL_ALLOCATION.get(risk, IDEAL_ALLOCATION["moderate"])
    for category, ideal_pct in ideal.items():
        current_pct = exposure.get(category, 0.0)
        deviation = current_pct - ideal_pct
        if abs(deviation) < ERROR_CATEGORY_DEVIATION_PP:
            continue
        candidates.append(
            InsightCandidate(
                id=f"error_detection.allocation.{category}",
                title=f"Alocacao critica em {category}",
                body=(
                    f"Exposicao a {category} em {current_pct}% "
                    f"(meta {ideal_pct}% para perfil {risk})."
                ),
                evidence=[
                    InsightEvidence(
                        label=f"Exposicao atual em {category}",
                        value=current_pct,
                        source=f"exposure.{category}",
                    ),
                    InsightEvidence(
                        label=f"Meta de {category} para perfil {risk}",
                        value=ideal_pct,
                        source=f"policy.ideal_allocation.{risk}.{category}",
                    ),
                    InsightEvidence(
                        label="Desvio critico em pontos percentuais",
                        value=round(deviation, 2),
                        source=f"derived.deviation.{category}",
                    ),
                    InsightEvidence(
                        label="Limiar critico de desvio",
                        value=ERROR_CATEGORY_DEVIATION_PP,
                        source="policy.error_category_deviation_pp",
                    ),
                ],
                action=InsightAction(
                    label=f"Abrir rebalanceador para {category} = {ideal_pct}%",
                    route="/rebalancer",
                    payload={"category": category, "target_pct": ideal_pct},
                    why=(
                        f"Alocacao em {category} desviou mais de "
                        f"{ERROR_CATEGORY_DEVIATION_PP}pp da meta do "
                        f"perfil {risk}."
                    ),
                ),
                fallback_rationale=(
                    f"A exposicao a {category} esta em {current_pct}%, ante "
                    f"meta de {ideal_pct}% para o perfil {risk}. O desvio "
                    f"supera o limiar critico de {ERROR_CATEGORY_DEVIATION_PP}pp."
                ),
            )
        )

    for asset in profile.portfolio.assets:
        if _normalize_category(asset.type) != "fii":
            continue
        metrics = asset.metrics or {}
        pvp = metrics.get("pvp_ratio")
        if pvp is None or pvp <= FII_PVP_ALERT:
            continue
        candidates.append(
            InsightCandidate(
                id=f"error_detection.fii_pvp.{asset.symbol}",
                title=f"FII {asset.symbol} com P/VP elevado",
                body=f"{asset.symbol} negocia com P/VP {pvp}.",
                evidence=[
                    InsightEvidence(
                        label=f"P/VP atual de {asset.symbol}",
                        value=pvp,
                        source=f"portfolio.asset.{asset.symbol}.metrics.pvp_ratio",
                    ),
                    InsightEvidence(
                        label="Limiar de alerta de P/VP",
                        value=FII_PVP_ALERT,
                        source="policy.fii_pvp_alert",
                    ),
                ],
                action=InsightAction(
                    label=f"Revisar {asset.symbol} no rebalanceador",
                    route="/rebalancer",
                    payload={"symbol": asset.symbol},
                    why=(
                        f"P/VP de {asset.symbol} passou do limiar de alerta "
                        f"de {FII_PVP_ALERT} — premio historicamente frio."
                    ),
                ),
                fallback_rationale=(
                    f"{asset.symbol} esta com P/VP de {pvp}, acima do limiar "
                    f"de {FII_PVP_ALERT} usado pela estrategia de FIIs. "
                    "Premio alto historicamente pesa no retorno futuro."
                ),
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# recommendations (rebalance moves)
# ---------------------------------------------------------------------------


def build_recommendations_evidence(profile: UserProfile) -> List[InsightCandidate]:
    """
    TRA-135: recommendations migrado.

    O legado (`rebalancing.top_moves`) vinha como frases livres tipo
    "Reduzir Acoes (70% -> 30%)". Aqui viram insights com evidencia
    numerica (atual, meta, desvio, direcao) e acao com rota do
    rebalanceador.

    IMPORTANTE: NAO emitimos COMPRA/HOLD/VENDA por ativo — isso saiu com
    TRA-53. Recomendacao aqui e sempre movimento de CATEGORIA, coerente
    com o produto ("nao indica ativos").
    """

    exposure = exposure_by_category(profile)
    risk = (profile.risk_profile or "moderate").lower()
    ideal = IDEAL_ALLOCATION.get(risk, IDEAL_ALLOCATION["moderate"])
    candidates: List[InsightCandidate] = []
    for category, ideal_pct in ideal.items():
        current_pct = exposure.get(category, 0.0)
        deviation = current_pct - ideal_pct
        if abs(deviation) < CATEGORY_DEVIATION_PP:
            continue
        direction = "Reduzir" if deviation > 0 else "Aumentar"
        candidates.append(
            InsightCandidate(
                id=f"recommendations.rebalance.{category}",
                title=f"{direction} {category}",
                body=(
                    f"{direction} exposicao a {category}: {current_pct}% "
                    f"vs meta {ideal_pct}% para perfil {risk}."
                ),
                evidence=[
                    InsightEvidence(
                        label=f"Exposicao atual em {category}",
                        value=current_pct,
                        source=f"exposure.{category}",
                    ),
                    InsightEvidence(
                        label=f"Meta de {category} para perfil {risk}",
                        value=ideal_pct,
                        source=f"policy.ideal_allocation.{risk}.{category}",
                    ),
                    InsightEvidence(
                        label="Desvio em pontos percentuais",
                        value=round(deviation, 2),
                        source=f"derived.deviation.{category}",
                    ),
                ],
                action=InsightAction(
                    label=f"{direction} {category} para {ideal_pct}%",
                    route="/rebalancer",
                    payload={
                        "category": category,
                        "target_pct": ideal_pct,
                        "direction": direction.lower(),
                    },
                    why=(
                        f"Movimento de categoria — o produto nao indica "
                        f"ativos, so ajuda a manter a alocacao proxima da "
                        f"meta do perfil {risk}."
                    ),
                ),
                fallback_rationale=(
                    f"A alocacao em {category} esta em {current_pct}%, ante "
                    f"meta de {ideal_pct}% para o perfil {risk}. Recomendacao "
                    f"e {direction.lower()} a categoria — nunca um ativo "
                    "especifico."
                ),
            )
        )
    return candidates


PRODUCERS = {
    "smart_feed": build_smart_feed_evidence,
    "opportunity_radar": build_opportunity_radar_evidence,
    "error_detection": build_error_detection_evidence,
    "recommendations": build_recommendations_evidence,
}
