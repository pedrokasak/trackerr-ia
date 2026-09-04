"""
Pipeline de insights com profundidade (TRA-56).

O LLM antes emitia frases curtas ("Reduza exposicao a cripto") sem dizer
POR QUE, com base em QUE dado, e o que FAZER a seguir. E ja alucinou
numero-alvo (TRA-55). Este modulo inverte a divisao de trabalho:

  * Numeros vem do codigo (evidencia deterministica).
  * Confianca vem do codigo (frescor, contagem de fontes, cobertura).
  * Acao (rota + payload) vem do codigo.
  * LLM SO escreve rationale/label a partir da evidencia — nunca inventa
    numero. O parser rejeita qualquer numero na resposta que nao apareca
    na evidencia; em caso de rejeicao, fallback deterministico.

Isso mantem o produto dentro da promessa publica (nao e consultoria de
investimento — descreve fato, nao emite ordem) e faz o insight ser
auditavel de ponta a ponta.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from benchmark.providers.base import LLMProvider
from models.models import (
    Insight,
    InsightAction,
    InsightConfidence,
    InsightEvidence,
    InsightSource,
    UserProfile,
)


# ---------------------------------------------------------------------------
# Evidencia deterministica
# ---------------------------------------------------------------------------

CATEGORY_ALIASES = {
    "stock": "acoes",
    "acao": "acoes",
    "acoes": "acoes",
    "fii": "fii",
    "etf": "etf",
    "crypto": "cripto",
    "cripto": "cripto",
    "rf": "renda_fixa",
    "renda_fixa": "renda_fixa",
    "fixed_income": "renda_fixa",
}

# Meta de alocacao por perfil de risco. Nao e "recomendacao personalizada",
# e a mesma referencia ja usada em `prepare_analysis_prompt`.
IDEAL_ALLOCATION = {
    "conservative": {"renda_fixa": 60, "acoes": 15, "fii": 15, "etf": 5, "cripto": 0},
    "moderate": {"renda_fixa": 40, "acoes": 30, "fii": 15, "etf": 10, "cripto": 5},
    "aggressive": {"renda_fixa": 20, "acoes": 45, "fii": 15, "etf": 10, "cripto": 10},
}

# Limite de concentracao em um unico ativo, em % do patrimonio.
SINGLE_ASSET_CONCENTRATION_LIMIT_PCT = 30.0
# Desvio (em pontos percentuais) acima do qual disparamos um insight de
# rebalanceamento por categoria.
CATEGORY_DEVIATION_PP = 10.0


def _normalize_category(raw: Optional[str]) -> str:
    return CATEGORY_ALIASES.get((raw or "").lower(), "outros")


def exposure_by_category(profile: UserProfile) -> Dict[str, float]:
    """Percentual do patrimonio por categoria. Total pode nao somar 100 quando
    a `total_value` do portfolio nao bate com a soma real das posicoes — nao
    corrigimos aqui, o dado vira como o server entregou."""

    total = profile.portfolio.total_value or sum(
        (a.quantity or 0) * (a.current_price or 0) for a in profile.portfolio.assets
    )
    if total <= 0:
        return {}
    exposure: Dict[str, float] = {}
    for asset in profile.portfolio.assets:
        cat = _normalize_category(asset.type)
        value = (asset.quantity or 0) * (asset.current_price or 0)
        exposure[cat] = exposure.get(cat, 0.0) + value
    return {cat: round(v / total * 100, 2) for cat, v in exposure.items()}


def top_asset_concentration(profile: UserProfile) -> Optional[Dict[str, Any]]:
    total = profile.portfolio.total_value or sum(
        (a.quantity or 0) * (a.current_price or 0) for a in profile.portfolio.assets
    )
    if total <= 0 or not profile.portfolio.assets:
        return None
    top = max(
        profile.portfolio.assets,
        key=lambda a: (a.quantity or 0) * (a.current_price or 0),
    )
    value = (top.quantity or 0) * (top.current_price or 0)
    if value <= 0:
        return None
    return {"symbol": top.symbol, "pct": round(value / total * 100, 2)}


@dataclass
class InsightCandidate:
    """
    Insight ainda sem prosa do LLM. Carrega tudo que o codigo ja sabe:
    identidade, titulo curto, evidencia, acao e fallback.

    O `fallback_rationale` e usado quando o LLM devolve texto com numero
    inventado, ou quando o LLM falha. NAO e placeholder: e a versao "so
    codigo" do rationale, montada a partir dos rotulos da evidencia.
    """

    id: str
    title: str
    body: str
    evidence: List[InsightEvidence]
    action: Optional[InsightAction]
    ideal_pct: Optional[float] = None
    fallback_rationale: str = ""


def build_deterministic_evidence(profile: UserProfile) -> List[InsightCandidate]:
    """
    Deriva candidatos a insight direto da carteira. Cada regra e simples e
    explicita — sem heuristica opaca. Novas regras se somam aqui; a divisao
    de trabalho com o LLM nao muda.
    """

    candidates: List[InsightCandidate] = []
    exposure = exposure_by_category(profile)
    top = top_asset_concentration(profile)
    risk = (profile.risk_profile or "moderate").lower()
    ideal = IDEAL_ALLOCATION.get(risk, IDEAL_ALLOCATION["moderate"])

    if top and top["pct"] >= SINGLE_ASSET_CONCENTRATION_LIMIT_PCT:
        symbol = top["symbol"]
        pct = top["pct"]
        candidates.append(
            InsightCandidate(
                id=f"concentration.{symbol}",
                title=f"Concentracao alta em {symbol}",
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
                    label=f"Abrir rebalanceador com meta de {symbol} <= {SINGLE_ASSET_CONCENTRATION_LIMIT_PCT}%",
                    route="/rebalancer",
                    payload={"symbol": symbol, "target_max_pct": SINGLE_ASSET_CONCENTRATION_LIMIT_PCT},
                    why=f"{symbol} passou do limite de concentracao por ativo unico.",
                ),
                fallback_rationale=(
                    f"{symbol} responde por {pct}% da carteira, acima do limite de "
                    f"{SINGLE_ASSET_CONCENTRATION_LIMIT_PCT}%. Concentracao alta em "
                    "um unico ativo aumenta o impacto de uma queda especifica."
                ),
            )
        )

    for category, ideal_pct in ideal.items():
        current_pct = exposure.get(category, 0.0)
        deviation = current_pct - ideal_pct
        if abs(deviation) < CATEGORY_DEVIATION_PP:
            continue
        direction = "reduzir" if deviation > 0 else "aumentar"
        candidates.append(
            InsightCandidate(
                id=f"rebalance.{category}",
                title=f"Ajuste em {category}",
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
                        label="Desvio em pontos percentuais",
                        value=round(deviation, 2),
                        source=f"derived.deviation.{category}",
                    ),
                ],
                action=InsightAction(
                    label=f"Abrir rebalanceador com meta {category} = {ideal_pct}%",
                    route="/rebalancer",
                    payload={"category": category, "target_pct": ideal_pct},
                    why=f"Alocacao em {category} esta fora da meta do perfil {risk}.",
                ),
                ideal_pct=float(ideal_pct),
                fallback_rationale=(
                    f"A exposicao a {category} esta em {current_pct}%, enquanto a "
                    f"meta do perfil {risk} e {ideal_pct}%. Aproximar da meta "
                    f"pede {direction} a posicao."
                ),
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# Confianca
# ---------------------------------------------------------------------------

FRESHNESS_WINDOW_DAYS = 90
MIN_SOURCES_FOR_FULL_CONFIDENCE = 4
MIN_EVIDENCE_FOR_FULL_CONFIDENCE = 3


def _bucket(value: float) -> str:
    if value < 0.5:
        return "baixa"
    if value < 0.75:
        return "media"
    return "alta"


def compute_confidence(
    *,
    evidence_count: int,
    source_count: int,
    data_freshness_days: Optional[int],
) -> InsightConfidence:
    """
    Confianca a partir de tres fatores independentes:

      * frescor: 1.0 quando <=1d, decai linear ate 0.0 em FRESHNESS_WINDOW_DAYS,
        0.5 quando nao ha info (nao da pra afirmar frescor).
      * fontes: min(1, n/MIN_SOURCES_FOR_FULL_CONFIDENCE).
      * evidencia: min(1, n/MIN_EVIDENCE_FOR_FULL_CONFIDENCE).

    O bucket usa os limiares do issue TRA-56: <0.5 baixa, 0.5-0.75 media,
    >=0.75 alta. O `reason` cita explicitamente os numeros de entrada — se
    a UI mostrar so o bucket, o auditor ainda enxerga o que gerou.
    """

    if data_freshness_days is None:
        freshness_factor = 0.5
        freshness_note = "frescor indefinido"
    else:
        clamped = max(0, min(data_freshness_days, FRESHNESS_WINDOW_DAYS))
        freshness_factor = 1.0 - (clamped / FRESHNESS_WINDOW_DAYS)
        freshness_note = f"dados dos ultimos {data_freshness_days}d"

    source_factor = min(1.0, source_count / MIN_SOURCES_FOR_FULL_CONFIDENCE)
    evidence_factor = min(1.0, evidence_count / MIN_EVIDENCE_FOR_FULL_CONFIDENCE)

    value = round((freshness_factor + source_factor + evidence_factor) / 3, 3)
    reason = (
        f"{freshness_note}, {source_count} fontes, {evidence_count} evidencias"
    )
    return InsightConfidence(value=value, bucket=_bucket(value), reason=reason)


# ---------------------------------------------------------------------------
# Prompt + parser
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


class NumberHallucinationError(ValueError):
    """
    Resposta do LLM contem numero que nao aparece na evidencia. Sinaliza
    alucinacao (contexto do TRA-55) — o chamador cai para o fallback
    deterministico em vez de propagar o texto.
    """


def _numbers_in(text: str) -> List[str]:
    return [n.replace(",", ".") for n in _NUMBER_RE.findall(text or "")]


def _evidence_number_set(candidate: InsightCandidate) -> set[str]:
    numbers: set[str] = set()
    for ev in candidate.evidence:
        for n in _numbers_in(str(ev.value)):
            numbers.add(n)
            # Aceita "30" quando a evidencia guarda "30.0" (e vice-versa) sem
            # exigir do LLM formatacao especifica.
            try:
                numbers.add(str(int(float(n))))
                numbers.add(f"{float(n):.1f}")
                numbers.add(f"{float(n):.2f}")
            except ValueError:
                pass
    return numbers


def parse_llm_insight_response(
    raw: Dict[str, Any], candidates: List[InsightCandidate]
) -> Dict[str, Dict[str, str]]:
    """
    Extrai `rationale` e `action_label` por insight_id do payload do LLM,
    validando que nenhum numero citado esta fora da evidencia daquele
    insight. Levanta NumberHallucinationError na primeira infracao — quem
    chama cai pro fallback com registro em log.
    """

    payload: Any = raw
    if isinstance(raw, dict) and "insights" not in raw:
        # Alguns providers embrulham como {"answer": "{...}"} ou devolvem
        # JSON dentro de string. Tentamos os dois com tolerancia.
        candidate_text = raw.get("raw_response") or raw.get("answer")
        if isinstance(candidate_text, str):
            try:
                payload = json.loads(candidate_text)
            except json.JSONDecodeError:
                payload = raw
    if not isinstance(payload, dict):
        raise NumberHallucinationError("resposta do LLM sem estrutura esperada")

    items = payload.get("insights") or []
    if not isinstance(items, list):
        raise NumberHallucinationError("campo 'insights' do LLM nao e lista")

    by_id = {c.id: c for c in candidates}
    result: Dict[str, Dict[str, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        insight_id = str(item.get("id", "")).strip()
        candidate = by_id.get(insight_id)
        if candidate is None:
            continue
        rationale = str(item.get("rationale", "")).strip()
        action_label = str(item.get("action_label", "")).strip()

        allowed = _evidence_number_set(candidate)
        for n in _numbers_in(rationale):
            if n not in allowed:
                raise NumberHallucinationError(
                    f"insight {insight_id}: numero '{n}' fora da evidencia"
                )
        result[insight_id] = {"rationale": rationale, "action_label": action_label}
    return result


def build_insights_prompt(candidates: List[InsightCandidate]) -> str:
    """
    Prompt curto e restrito. O LLM recebe a evidencia ja fechada e so
    escreve prosa. `title` e `action.label` deterministicos ficam
    disponiveis so como contexto, pra manter tom consistente — se o
    modelo tentar melhorar `action.label`, o parser aceita e o codigo
    ainda mantem a rota/payload determinados.
    """

    if not candidates:
        return ""
    items_json = json.dumps(
        [
            {
                "id": c.id,
                "title": c.title,
                "evidence": [
                    {"label": e.label, "value": e.value} for e in c.evidence
                ],
                "default_action_label": c.action.label if c.action else None,
            }
            for c in candidates
        ],
        ensure_ascii=False,
        indent=2,
    )
    return f"""
Voce escreve a explicacao curta ("por que") e o rotulo de acao de insights de
carteira do Trackerr. O calculo ja foi feito no codigo — voce so redige.

REGRAS OBRIGATORIAS:
- Escreva em portugues do Brasil.
- Use APENAS os numeros que aparecem em `evidence.value`. Nao invente
  numero, percentual, meta ou preco.
- Nao emita ordem de compra ou venda; descreva o fato, cite o limite ou meta
  e diga qual acao geral resolve.
- `rationale`: 2 ou 3 frases.
- `action_label`: uma frase curta comecando por verbo.

ENTRADA:
{items_json}

Retorne APENAS JSON no formato:
{{
  "insights": [
    {{"id": "...", "rationale": "...", "action_label": "..."}}
  ]
}}
"""


# ---------------------------------------------------------------------------
# Orquestrador
# ---------------------------------------------------------------------------


@dataclass
class _NarratedInsight:
    rationale: str
    action_label: str


class InsightsService:
    def __init__(
        self,
        llm_provider: LLMProvider,
        *,
        logger=None,
    ) -> None:
        self._llm = llm_provider
        self._logger = logger

    async def generate(
        self,
        profile: UserProfile,
        *,
        data_freshness_days: Optional[int] = None,
        sources: Optional[List[InsightSource]] = None,
        producer: Optional[Callable[[UserProfile], List[InsightCandidate]]] = None,
    ) -> List[Insight]:
        # TRA-135: `producer` permite migrar producers legados de
        # /api/hybrid-analysis (opportunity_radar, error_detection,
        # smart_feed, recommendations) reaproveitando o mesmo pipeline
        # — evidencia deterministica, narracao com guardrail anti-alucinacao,
        # confianca calculada, acao com rota. Default mantem BC do TRA-56.
        candidate_builder = producer or build_deterministic_evidence
        candidates = candidate_builder(profile)
        if not candidates:
            return []

        sources = sources or []
        narrated = await self._narrate(candidates)

        insights: List[Insight] = []
        for c in candidates:
            n = narrated.get(
                c.id,
                _NarratedInsight(
                    rationale=c.fallback_rationale,
                    action_label=c.action.label if c.action else "",
                ),
            )
            confidence = compute_confidence(
                evidence_count=len(c.evidence),
                source_count=len(sources),
                data_freshness_days=data_freshness_days,
            )
            action = c.action
            if action and n.action_label and n.action_label != action.label:
                # LLM pode refinar o rotulo humano; rota e payload continuam
                # sendo do codigo — nunca aceitar rota/payload do modelo.
                action = InsightAction(
                    label=n.action_label,
                    route=action.route,
                    payload=action.payload,
                    why=action.why,
                )
            insights.append(
                Insight(
                    id=c.id,
                    title=c.title,
                    body=c.body,
                    rationale=n.rationale or c.fallback_rationale,
                    evidence=c.evidence,
                    confidence=confidence,
                    action=action,
                    sources=sources,
                )
            )
        return insights

    async def _narrate(
        self, candidates: List[InsightCandidate]
    ) -> Dict[str, _NarratedInsight]:
        prompt = build_insights_prompt(candidates)
        if not prompt:
            return {}
        try:
            raw = await self._llm.analyze(prompt)
            parsed = parse_llm_insight_response(raw, candidates)
        except NumberHallucinationError as exc:
            if self._logger:
                self._logger.warning(
                    "insight LLM devolveu numero fora da evidencia: %s", exc
                )
            return {}
        except Exception as exc:  # pragma: no cover - falha de rede/provider
            if self._logger:
                self._logger.error("falha ao chamar LLM para insights: %s", exc)
            return {}
        return {
            k: _NarratedInsight(
                rationale=v.get("rationale", ""), action_label=v.get("action_label", "")
            )
            for k, v in parsed.items()
        }
