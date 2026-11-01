# -*- coding: utf-8 -*-
"""
ADR-004, 9.2: proveniência das saídas de IA.

Prova o contrato: o bloco diz quem respondeu de fato (inclusive quando o
fallback respondeu), de que entrada saiu e o que o guarda fez, e carrega só
ids e hashes — nunca o texto do prompt, da carteira ou da resposta.
"""

import json
from typing import Any, Dict

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from benchmark.providers.base import LLMProvider
from benchmark.providers.fallback_provider import FallbackLLMProvider
from insights.service import InsightsService
from models.models import Asset, InsightSource, Portfolio, UserProfile
from provenance import (
    ProvenanceSource,
    canonical_json,
    data_window_of,
    served_by_of,
    sha256_json,
    sha256_text,
)


class _LLM(LLMProvider):
    def __init__(self, name: str, response=None, error: Exception | None = None):
        self._name = name
        self._model = f"{name}-model"
        self._response = response if response is not None else {"insights": []}
        self._error = error

    async def analyze(self, prompt: str) -> Dict[str, Any]:
        if self._error:
            raise self._error
        return self._response

    @property
    def provider_name(self) -> str:
        return self._name


def _profile() -> UserProfile:
    portfolio = Portfolio(
        assets=[
            Asset(symbol="BTC", type="cripto", quantity=1, current_price=500.0),
            Asset(symbol="PETR4", type="stock", quantity=10, current_price=30.0),
            Asset(symbol="CDB", type="renda_fixa", quantity=1, current_price=200.0),
        ],
        total_value=1000.0,
    )
    return UserProfile(user_id="u1", risk_profile="moderate", portfolio=portfolio)


def test_json_canonico_e_estavel_e_o_hash_nao_depende_da_ordem():
    assert canonical_json({"b": 1, "a": [2, {"z": 1, "y": 2}]}) == (
        '{"a":[2,{"y":2,"z":1}],"b":1}'
    )
    assert sha256_json({"x": 1, "y": 2}) == sha256_json({"y": 2, "x": 1})
    assert sha256_text("a") != sha256_text("b")


def test_janela_de_dados_e_a_menor_e_a_maior_data_das_fontes():
    window = data_window_of(
        [
            ProvenanceSource(as_of="2026-10-24"),
            ProvenanceSource(as_of="2026-10-18"),
            ProvenanceSource(),
        ]
    )
    assert (window.from_, window.to) == ("2026-10-18", "2026-10-24")


def test_sem_data_nas_fontes_a_janela_fica_vazia():
    window = data_window_of([ProvenanceSource(source_id="BTC")])
    assert (window.from_, window.to) == (None, None)


@pytest.mark.asyncio
async def test_fallback_registra_quem_respondeu_de_fato():
    primary = _LLM("gemini", error=HTTPException(status_code=503, detail="fora"))
    fallback = _LLM("groq", response={"ok": True})
    provider = FallbackLLMProvider(primary, fallback)

    await provider.analyze("prompt")

    # `provider_name` continua sendo o primário; `served_by` diz a verdade.
    assert provider.provider_name == "gemini"
    assert provider.served_by() == {"provider": "groq", "model": "groq-model"}


@pytest.mark.asyncio
async def test_fallback_volta_a_dizer_primario_quando_ele_responde():
    primary = _LLM("gemini", response={"ok": True})
    provider = FallbackLLMProvider(primary, _LLM("groq"))

    await provider.analyze("prompt")

    assert provider.served_by() == {"provider": "gemini", "model": "gemini-model"}


def test_served_by_aceita_provider_sem_o_metodo():
    class _Duck:
        provider_name = "duck"

    assert served_by_of(_Duck()) == {"provider": "duck", "model": None}


@pytest.mark.asyncio
async def test_insights_trazem_proveniencia_sem_texto_nem_dado_pessoal():
    llm = _LLM("gemini")
    service = InsightsService(llm_provider=llm)
    sources = [
        InsightSource(
            source_type="portfolio_position", source_id="BTC", as_of="2026-10-24"
        )
    ]

    insights, provenance = await service.generate_with_provenance(
        _profile(), data_freshness_days=5, sources=sources
    )

    assert provenance is not None
    assert provenance.generation_id.startswith("gen_")
    assert provenance.kind == "insight"
    assert provenance.prompt.id == "insights.narration"
    assert provenance.model.served_provider == "gemini"
    assert provenance.model.model == "gemini-model"
    assert provenance.subject_ids == [i.id for i in insights]
    assert provenance.output_sha256 == sha256_json(insights)
    assert provenance.inputs.data_window.to == "2026-10-24"
    assert provenance.inputs.data_freshness_days == 5
    assert provenance.confidence is not None

    dumped = json.dumps(provenance.model_dump(by_alias=True))
    # Só ids e hashes: nada do que o usuário ou o modelo escreveram.
    assert "PETR4" not in dumped
    assert "Voce escreve" not in dumped  # texto do prompt
    assert "rationale" not in dumped


@pytest.mark.asyncio
async def test_guarda_ok_e_texto_do_modelo_nao_conta_como_fallback():
    llm = _LLM("gemini")
    service = InsightsService(llm_provider=llm)
    first = (await service.generate(_profile()))[0]
    llm._response = {
        "insights": [
            {
                "id": first.id,
                "rationale": "Texto curto sem numero algum.",
                "action_label": "Rever",
            }
        ]
    }

    insights, provenance = await service.generate_with_provenance(_profile())

    assert provenance.checks.guard == "ok"
    assert provenance.checks.fallback_count == len(insights) - 1


@pytest.mark.asyncio
async def test_numero_inventado_aparece_na_proveniencia_e_conta_fallback():
    llm = _LLM(
        "gemini",
        response={
            "insights": [
                {
                    "id": "concentration.BTC",
                    "rationale": "BTC vale R$ 9999.",
                    "action_label": "x",
                }
            ]
        },
    )
    service = InsightsService(llm_provider=llm)

    insights, provenance = await service.generate_with_provenance(_profile())

    assert provenance.checks.guard == "number_hallucination"
    assert provenance.checks.fallback_count == len(insights)


@pytest.mark.asyncio
async def test_falha_do_provider_aparece_como_llm_error():
    llm = _LLM("gemini", error=RuntimeError("timeout"))
    service = InsightsService(llm_provider=llm)

    insights, provenance = await service.generate_with_provenance(_profile())

    assert insights, "o insight deterministico continua saindo"
    assert provenance.checks.guard == "llm_error"
    assert provenance.checks.fallback_count == len(insights)


@pytest.mark.asyncio
async def test_sem_candidato_nao_ha_chamada_nem_proveniencia():
    service = InsightsService(llm_provider=_LLM("gemini"))
    insights, provenance = await service.generate_with_provenance(
        _profile(), producer=lambda profile: []
    )

    assert insights == []
    assert provenance is None


@pytest.mark.asyncio
async def test_mesma_entrada_da_mesmo_hash_e_geracao_nova_da_outro_id():
    service = InsightsService(llm_provider=_LLM("gemini"))
    _, a = await service.generate_with_provenance(_profile())
    _, b = await service.generate_with_provenance(_profile())

    assert a.inputs.portfolio_snapshot_sha256 == b.inputs.portfolio_snapshot_sha256
    assert a.inputs.evidence_sha256 == b.inputs.evidence_sha256
    assert a.prompt.sha256 == b.prompt.sha256
    assert a.generation_id != b.generation_id


def test_endpoint_de_insights_devolve_a_proveniencia(monkeypatch):
    from benchmark.providers import factory as factory_module
    from main import app

    monkeypatch.setattr(
        factory_module.LLMFactory, "get_provider", lambda: _LLM("gemini")
    )
    client = TestClient(app)
    payload = {"user_profile": _profile().model_dump(mode="json")}

    response = client.post("/api/insights", json=payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["provenance"]) == 1
    block = body["provenance"][0]
    assert block["prompt"]["id"] == "insights.narration"
    assert block["model"]["served_provider"] == "gemini"
    # O alias `from` chega ao server como o nome que o contrato promete.
    assert "from" in block["inputs"]["data_window"]
