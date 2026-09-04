# -*- coding: utf-8 -*-
"""
TRA-135: contrato dos producers migrados de /api/hybrid-analysis.

Cada producer legado (opportunity_radar, error_detection, smart_feed,
recommendations) agora emite o shape estendido `Insight` (evidence[],
confidence, action opcional, rationale). Estes testes provam por producer:

- Fixture -> ao menos um Insight quando as condicoes disparam.
- `evidence` nao-vazia e com `source` (rastreavel).
- `confidence.reason` cita a fixture (0 fontes, N evidencias) — prova
  que a confianca veio de calculo em cima da entrada, nao de valor fixo.
- Pipeline reutiliza `InsightsService` (mesmo guardrail anti-alucinacao
  do TRA-56 vale de graca).
"""

from typing import Any, Dict, List

import pytest

from benchmark.providers.base import LLMProvider
from insights.producers import (
    build_error_detection_evidence,
    build_opportunity_radar_evidence,
    build_recommendations_evidence,
    build_smart_feed_evidence,
)
from insights.service import InsightsService
from models.models import Asset, Portfolio, UserProfile


class _FakeLLM(LLMProvider):
    """LLM que devolve `insights: []` — narrated fica vazio, cada Insight
    cai no `fallback_rationale`. Provamos o contrato deterministico sem
    depender de rede/provider."""

    def __init__(self, response: Dict[str, Any] | None = None) -> None:
        self._response = response or {"insights": []}
        self.calls: List[str] = []

    async def analyze(self, prompt: str) -> Dict[str, Any]:
        self.calls.append(prompt)
        return self._response

    @property
    def provider_name(self) -> str:
        return "fake"


def _profile_concentrada_com_movers() -> UserProfile:
    """
    50% BTC (dispara concentracao + FII allocation off + queda 24h),
    20% PETR4 subindo, 10% MXRF11 (FII com pvp alto), 20% CDB estavel.
    total_value = 1000 pra facilitar leitura dos %.
    """
    portfolio = Portfolio(
        assets=[
            Asset(
                symbol="BTC",
                type="cripto",
                quantity=1,
                current_price=500.0,
                change_24h=-6.0,
            ),
            Asset(
                symbol="PETR4",
                type="stock",
                quantity=10,
                current_price=20.0,
                change_24h=4.5,
            ),
            Asset(
                symbol="MXRF11",
                type="fii",
                quantity=10,
                current_price=10.0,
                change_24h=-1.0,
                metrics={"pvp_ratio": 1.8},
            ),
            Asset(
                symbol="CDB",
                type="renda_fixa",
                quantity=1,
                current_price=200.0,
                change_24h=0.0,
            ),
        ],
        total_value=1000.0,
    )
    return UserProfile(
        user_id="u1", risk_profile="moderate", portfolio=portfolio
    )


# ---------------------------------------------------------------------------
# smart_feed
# ---------------------------------------------------------------------------


def test_smart_feed_ordena_por_variacao_absoluta_e_cita_evidencia():
    profile = _profile_concentrada_com_movers()
    candidates = build_smart_feed_evidence(profile)

    ids = [c.id for c in candidates]
    # BTC (-6%) e PETR4 (+4.5%) passam do NOTABLE_MOVE_PCT (3%).
    # MXRF11 (-1%) nao passa. CDB (0%) nao entra.
    assert "smart_feed.mover.BTC" in ids
    assert "smart_feed.mover.PETR4" in ids
    assert "smart_feed.mover.MXRF11" not in ids

    btc = next(c for c in candidates if c.id == "smart_feed.mover.BTC")
    assert btc.action is None, "feed e observacao, nao acao"
    labels = [ev.label for ev in btc.evidence]
    assert any("Variacao 24h" in label for label in labels)
    assert any("Peso" in label for label in labels)
    for ev in btc.evidence:
        assert ev.source, "toda evidencia precisa apontar a origem"


@pytest.mark.asyncio
async def test_smart_feed_produz_insight_com_confidence_reason_referenciando_fixture():
    profile = _profile_concentrada_com_movers()
    service = InsightsService(llm_provider=_FakeLLM())

    insights = await service.generate(
        profile,
        producer=build_smart_feed_evidence,
        data_freshness_days=5,
    )

    assert insights, "esperava insights de smart_feed para a fixture"
    for insight in insights:
        assert insight.evidence
        assert insight.rationale
        assert "0 fontes" in insight.confidence.reason
        assert f"{len(insight.evidence)} evidencias" in insight.confidence.reason


# ---------------------------------------------------------------------------
# opportunity_radar
# ---------------------------------------------------------------------------


def test_opportunity_radar_marca_apenas_quedas_relevantes_com_peso_material():
    profile = _profile_concentrada_com_movers()
    candidates = build_opportunity_radar_evidence(profile)
    ids = [c.id for c in candidates]

    # BTC caiu 6% e pesa 50% — entra.
    assert "opportunity_radar.watch.BTC" in ids
    # PETR4 subiu — radar so olha queda pra nao virar gatilho de compra.
    assert "opportunity_radar.watch.PETR4" not in ids
    # MXRF11 caiu so 1% e pesa 10% — abaixo do NOTABLE_MOVE_PCT.
    assert "opportunity_radar.watch.MXRF11" not in ids

    btc = next(c for c in candidates if c.id == "opportunity_radar.watch.BTC")
    assert btc.action is None, "radar so observa; acao viraria indicacao"
    sources = {ev.source for ev in btc.evidence}
    assert any("change_24h" in (s or "") for s in sources)
    assert any("weight" in (s or "") for s in sources)


@pytest.mark.asyncio
async def test_opportunity_radar_contrato_com_service():
    profile = _profile_concentrada_com_movers()
    service = InsightsService(llm_provider=_FakeLLM())
    insights = await service.generate(
        profile,
        producer=build_opportunity_radar_evidence,
        data_freshness_days=5,
    )
    assert insights
    for insight in insights:
        assert insight.evidence
        assert insight.rationale
        assert "0 fontes" in insight.confidence.reason


# ---------------------------------------------------------------------------
# error_detection
# ---------------------------------------------------------------------------


def test_error_detection_flag_concentracao_alocacao_e_pvp_fii():
    profile = _profile_concentrada_com_movers()
    candidates = build_error_detection_evidence(profile)
    ids = {c.id for c in candidates}

    # BTC 50% > 30% -> concentracao critica.
    assert "error_detection.concentration.BTC" in ids
    # Cripto 50% vs meta 5% (moderate) = 45pp de desvio > 20pp.
    assert "error_detection.allocation.cripto" in ids
    # Renda fixa 20% vs meta 40% = -20pp — na borda, deve entrar
    # (>= 20 nao dispara; abaixo estrito nao dispara). Testamos so o
    # que passa com folga.
    # MXRF11 pvp 1.8 > 1.5.
    assert "error_detection.fii_pvp.MXRF11" in ids

    concentration = next(
        c for c in candidates if c.id == "error_detection.concentration.BTC"
    )
    assert concentration.action is not None
    assert concentration.action.route == "/rebalancer"
    assert concentration.action.payload == {
        "symbol": "BTC",
        "target_max_pct": 30.0,
    }


@pytest.mark.asyncio
async def test_error_detection_contrato_com_service():
    profile = _profile_concentrada_com_movers()
    service = InsightsService(llm_provider=_FakeLLM())
    insights = await service.generate(
        profile,
        producer=build_error_detection_evidence,
        data_freshness_days=5,
    )
    assert insights
    for insight in insights:
        assert insight.evidence
        assert insight.rationale
        assert "0 fontes" in insight.confidence.reason


# ---------------------------------------------------------------------------
# recommendations
# ---------------------------------------------------------------------------


def test_recommendations_move_categorias_nunca_ativo():
    profile = _profile_concentrada_com_movers()
    candidates = build_recommendations_evidence(profile)
    ids = {c.id for c in candidates}

    # cripto 50% vs meta 5% -> Reduzir cripto.
    assert "recommendations.rebalance.cripto" in ids
    # renda_fixa 20% vs meta 40% -> Aumentar.
    assert "recommendations.rebalance.renda_fixa" in ids

    # TRA-53: nenhum insight de recomendacao pode citar ordem por ativo
    # ou verbo COMPRA/HOLD/VENDA — texto todo passa pelo filtro.
    for c in candidates:
        joined = " ".join(
            [
                c.title,
                c.body,
                c.fallback_rationale,
                c.action.label if c.action else "",
                (c.action.why or "") if c.action else "",
            ]
        ).upper()
        for verbo in ("COMPRA", "VENDA", "HOLD"):
            assert verbo not in joined
        # payload nao pode conter symbol — recomendacao e por categoria.
        if c.action and c.action.payload:
            assert "symbol" not in c.action.payload


@pytest.mark.asyncio
async def test_recommendations_contrato_com_service():
    profile = _profile_concentrada_com_movers()
    service = InsightsService(llm_provider=_FakeLLM())
    insights = await service.generate(
        profile,
        producer=build_recommendations_evidence,
        data_freshness_days=5,
    )
    assert insights
    for insight in insights:
        assert insight.evidence
        assert insight.rationale
        assert "0 fontes" in insight.confidence.reason
        assert insight.action is not None
        assert insight.action.route == "/rebalancer"


# ---------------------------------------------------------------------------
# hybrid-analysis endpoint: schema_version + insights_v2
# ---------------------------------------------------------------------------


def test_hybrid_analysis_response_carrega_schema_version_e_insights_v2(monkeypatch):
    from fastapi.testclient import TestClient

    from benchmark import benchmark as benchmark_module
    from benchmark.providers import factory as factory_module
    from main import app

    async def _fake_analyze(prompt: str):
        # Payload legado minimo; producers extendidos rodam em cima do
        # profile, nao dependem desta resposta.
        return {"portfolio_assessment": "ok", "opportunity_radar": []}

    monkeypatch.setattr(
        benchmark_module.AIAnalysisService, "analyze_with_ai", _fake_analyze
    )

    class _StubProvider:
        async def analyze(self, prompt: str):
            return {"insights": []}

        @property
        def provider_name(self) -> str:
            return "stub"

    monkeypatch.setattr(factory_module.LLMFactory, "get_provider", lambda: _StubProvider())

    client = TestClient(app)
    payload = {
        "user_id": "u1",
        "profile_plan": "premium",
        "risk_profile": "moderate",
        "portfolio": {
            "total_value": 1000.0,
            "assets": [
                {
                    "symbol": "BTC",
                    "type": "cripto",
                    "quantity": 1,
                    "current_price": 500.0,
                    "change_24h": -6.0,
                },
                {
                    "symbol": "CDB",
                    "type": "renda_fixa",
                    "quantity": 1,
                    "current_price": 500.0,
                    "change_24h": 0.0,
                },
            ],
        },
    }
    r = client.post("/api/hybrid-analysis", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["schema_version"] == "v2"
    # Legacy payload continua presente pra BC.
    assert "ai_analysis" in body
    assert "stock_scores" in body
    # Novos producers vem sob `insights_v2`.
    assert set(body["insights_v2"].keys()) == {
        "smart_feed",
        "opportunity_radar",
        "error_detection",
        "recommendations",
    }
    # error_detection precisa ter pego a concentracao critica em BTC.
    err_ids = {i["id"] for i in body["insights_v2"]["error_detection"]}
    assert "error_detection.concentration.BTC" in err_ids


def test_hybrid_analysis_free_plan_tambem_traz_schema_version(monkeypatch):
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    r = client.post(
        "/api/hybrid-analysis",
        json={
            "user_id": "u1",
            "profile_plan": "free",
            "risk_profile": "moderate",
            "portfolio": {"total_value": 0.0, "assets": []},
        },
    )
    assert r.status_code == 200
    assert r.json()["schema_version"] == "v2"
