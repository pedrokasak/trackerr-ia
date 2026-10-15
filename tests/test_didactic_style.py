"""Estilo didático por nível nos prompts que escrevem para o usuário (RF-6)."""

import asyncio
from typing import Any, Dict, List

import pytest

from benchmark.benchmark import AIAnalysisService
from benchmark.providers.base import LLMProvider
from fixed_income.verdict_service import FixedIncomeVerdictService
from insights.service import InsightsService, build_deterministic_evidence, build_insights_prompt
from models.models import Asset, FixedIncomeVerdictRequest, Portfolio, UserProfile
from style.didactic import didactic_rules, normalize_level
from tests.test_fixed_income_verdict import build_payload


@pytest.mark.parametrize(
    "value, expected",
    [
        ("iniciante", "iniciante"),
        ("beginner", "iniciante"),
        ("Intermediário", "intermediario"),
        ("experienced", "avancado"),
        ("avançado", "avancado"),
        (None, "intermediario"),
        ("qualquer", "intermediario"),
    ],
)
def test_normaliza_o_nivel_dos_dois_vocabularios(value, expected):
    assert normalize_level(value) == expected


def test_regras_comuns_e_especificas_por_nivel():
    beginner = didactic_rules("iniciante")
    advanced = didactic_rules("avancado")

    for rules in (beginner, advanced):
        assert "resposta direta" in rules
        assert "nunca uma ordem" in rules
    assert "nada de jargão" in beginner
    assert "sem analogias" in advanced
    assert "nada de jargão" not in advanced


def test_chat_leva_o_nivel_e_proibe_recomendar():
    prompt = AIAnalysisService.prepare_chat_prompt("Meu risco?", level="beginner")

    assert "nada de jargão" in prompt
    assert "Não recomende comprar ou vender" in prompt
    # Sem nível, o meio-termo.
    assert "definição curta" in AIAnalysisService.prepare_chat_prompt("Meu risco?")


def test_veredito_usa_o_nivel_sem_tirar_as_regras_fixas():
    prompt = FixedIncomeVerdictService.prepare_prompt(
        FixedIncomeVerdictRequest(**build_payload(level="avancado"))
    )

    assert "sem analogias" in prompt
    assert "Nunca recomende comprar" in prompt
    # As regras fixas vêm antes do estilo.
    assert prompt.index("REGRAS OBRIGATÓRIAS") < prompt.index("COMO EXPLICAR")


class _FakeLLM(LLMProvider):
    def __init__(self) -> None:
        self.calls: List[str] = []

    async def analyze(self, prompt: str) -> Dict[str, Any]:
        self.calls.append(prompt)
        return {"insights": []}

    @property
    def provider_name(self) -> str:
        return "fake"


def _profile() -> UserProfile:
    return UserProfile(
        user_id="u1",
        risk_profile="moderate",
        portfolio=Portfolio(
            assets=[
                Asset(symbol="BTC", type="cripto", quantity=1, current_price=500.0),
                Asset(symbol="PETR4", type="stock", quantity=10, current_price=30.0),
            ],
            total_value=800.0,
        ),
    )


def test_insights_passam_o_nivel_ate_o_prompt():
    llm = _FakeLLM()
    service = InsightsService(llm_provider=llm)

    asyncio.run(service.generate(_profile(), level="iniciante"))

    assert llm.calls and "nada de jargão" in llm.calls[0]
    candidates = build_deterministic_evidence(_profile())
    assert "definição curta" in build_insights_prompt(candidates)
