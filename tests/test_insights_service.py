# -*- coding: utf-8 -*-
"""
TRA-56: pipeline de insights com profundidade.

Prova o contrato: evidencia deterministica calculada do portfolio, LLM so
narra prosa a partir dessa evidencia, parser rejeita numero fora da
evidencia, confianca calculada em codigo com bucket e reason que citam a
contagem de entradas, acao com rota+payload deterministicos que sobrevivem
mesmo quando o LLM falha (fallback).
"""

from typing import Any, Dict, List

import pytest

from benchmark.providers.base import LLMProvider
from insights.service import (
    InsightsService,
    NumberHallucinationError,
    build_deterministic_evidence,
    compute_confidence,
    exposure_by_category,
    parse_llm_insight_response,
)
from models.models import Asset, InsightSource, Portfolio, UserProfile


def _profile(**overrides) -> UserProfile:
    portfolio = Portfolio(
        assets=[
            # 50% concentrado em BTC → dispara concentration.BTC (>= 30%).
            Asset(symbol="BTC", type="cripto", quantity=1, current_price=500.0),
            Asset(symbol="PETR4", type="stock", quantity=10, current_price=30.0),
            Asset(symbol="CDB", type="renda_fixa", quantity=1, current_price=200.0),
        ],
        total_value=1000.0,
    )
    kwargs = {"user_id": "u1", "risk_profile": "moderate", "portfolio": portfolio}
    kwargs.update(overrides)
    return UserProfile(**kwargs)


class _FakeLLM(LLMProvider):
    def __init__(self, response: Dict[str, Any]) -> None:
        self._response = response
        self.calls: List[str] = []

    async def analyze(self, prompt: str) -> Dict[str, Any]:
        self.calls.append(prompt)
        return self._response

    @property
    def provider_name(self) -> str:
        return "fake"


def test_exposure_by_category_normaliza_tipos():
    exposure = exposure_by_category(_profile())
    assert exposure["cripto"] == 50.0
    assert exposure["acoes"] == 30.0
    assert exposure["renda_fixa"] == 20.0


def test_evidencia_deterministica_cobre_concentracao_e_categoria():
    candidates = build_deterministic_evidence(_profile())
    ids = {c.id for c in candidates}

    # Concentracao em ativo unico dispara porque BTC = 50% >= 30%.
    assert "concentration.BTC" in ids
    concentration = next(c for c in candidates if c.id == "concentration.BTC")
    labels = [ev.label for ev in concentration.evidence]
    assert any("BTC" in label for label in labels)
    assert concentration.action is not None
    assert concentration.action.route == "/rebalancer"
    assert concentration.action.payload == {
        "symbol": "BTC",
        "target_max_pct": 30.0,
    }

    # Ajuste de categoria dispara para renda_fixa (20% vs meta 40%) e cripto
    # (50% vs meta 5%), ambos alem do desvio de 10pp.
    assert "rebalance.renda_fixa" in ids
    assert "rebalance.cripto" in ids


def test_confidence_bucket_e_reason_referenciam_entradas():
    baixa = compute_confidence(evidence_count=1, source_count=0, data_freshness_days=None)
    media = compute_confidence(evidence_count=2, source_count=2, data_freshness_days=15)
    alta = compute_confidence(evidence_count=3, source_count=4, data_freshness_days=1)

    assert baixa.bucket == "baixa"
    assert media.bucket == "media"
    assert alta.bucket == "alta"

    # Reason cita explicitamente os numeros de entrada — auditor consegue
    # reconstruir o calculo sem abrir codigo.
    assert "0 fontes" in baixa.reason
    assert "2 evidencias" in media.reason
    assert "1d" in alta.reason


def test_parser_aceita_numero_presente_na_evidencia():
    candidates = build_deterministic_evidence(_profile())
    concentration = next(c for c in candidates if c.id == "concentration.BTC")
    resposta = {
        "insights": [
            {
                "id": concentration.id,
                # 50 e 30 aparecem em evidence.value — ok.
                "rationale": "BTC representa 50% da carteira, acima do limite de 30%.",
                "action_label": "Reduzir posicao em BTC",
            }
        ]
    }
    parsed = parse_llm_insight_response(resposta, candidates)
    assert parsed[concentration.id]["rationale"].startswith("BTC representa")


def test_parser_rejeita_numero_inventado():
    """Guardrail principal do TRA-55 aplicado aqui: LLM nao pode chutar 8000."""
    candidates = build_deterministic_evidence(_profile())
    concentration = next(c for c in candidates if c.id == "concentration.BTC")
    resposta = {
        "insights": [
            {
                "id": concentration.id,
                "rationale": "BTC caiu para R$ 8000 e representa 50% da carteira.",
                "action_label": "Reduzir",
            }
        ]
    }
    with pytest.raises(NumberHallucinationError):
        parse_llm_insight_response(resposta, candidates)


@pytest.mark.asyncio
async def test_service_produz_dto_valido_com_evidence_e_confidence_reason():
    """
    Contrato end-to-end: entrada fixture -> DTO estendido valido, com evidence
    nao-vazia e `confidence.reason` citando o tamanho da fixture (0 fontes,
    N evidencias). Cobre o producer `insights` novo — os producers legados
    (opportunity_radar, error_detection, smart_feed, recommendations em
    /api/hybrid-analysis) continuam emitindo a forma antiga; migracao dos
    demais e follow-up.
    """
    llm = _FakeLLM({"insights": []})  # sem prosa: cai no fallback deterministico
    service = InsightsService(llm_provider=llm)

    insights = await service.generate(_profile(), data_freshness_days=5)

    assert insights, "esperava ao menos um insight para a fixture"
    for insight in insights:
        assert insight.title
        assert insight.body
        assert insight.rationale, "rationale nao pode ficar vazio nem no fallback"
        assert insight.evidence, "evidencia deve ser nao-vazia"
        assert insight.action is not None
        assert insight.action.route == "/rebalancer"
        # `sources` = 0 nesta fixture (nao mandamos RAG), entao a reason
        # tem que refletir isso — prova que confidence foi calculada a partir
        # do tamanho real da entrada, nao um valor fixo.
        assert "0 fontes" in insight.confidence.reason
        assert f"{len(insight.evidence)} evidencias" in insight.confidence.reason
        assert insight.confidence.bucket in {"baixa", "media", "alta"}
        assert 0.0 <= insight.confidence.value <= 1.0


@pytest.mark.asyncio
async def test_service_usa_fallback_quando_llm_alucina_numero():
    """LLM devolve numero fora da evidencia -> service cai no fallback e
    NENHUM insight some da resposta."""
    llm = _FakeLLM(
        {
            "insights": [
                {
                    "id": "concentration.BTC",
                    "rationale": "BTC vale R$ 9999 e 50% da carteira.",  # 9999 inventado
                    "action_label": "Reduzir",
                }
            ]
        }
    )
    service = InsightsService(llm_provider=llm)
    insights = await service.generate(_profile(), data_freshness_days=5)

    assert any(i.id == "concentration.BTC" for i in insights)
    concentration = next(i for i in insights if i.id == "concentration.BTC")
    assert "9999" not in concentration.rationale
    assert "30%" in concentration.rationale or "30.0" in concentration.rationale


@pytest.mark.asyncio
async def test_service_repassa_sources_do_rag():
    llm = _FakeLLM({"insights": []})
    service = InsightsService(llm_provider=llm)
    sources = [InsightSource(source_type="portfolio_position", source_id="BTC")]

    insights = await service.generate(
        _profile(), data_freshness_days=1, sources=sources
    )
    assert all(i.sources == sources for i in insights)
