from benchmark.benchmark import DigestNarrationService
from models.models import (
    PortfolioDigestFactsInput,
    DigestMoverInput,
    DigestWatchItemInput,
)


def build_facts(**overrides) -> PortfolioDigestFactsInput:
    defaults = dict(
        period_start="2026-08-11",
        period_end="2026-08-18",
        portfolio_value=10000.0,
        period_change_pct=5.0,
        period_change_abs=500.0,
        top_gainers=[DigestMoverInput(symbol="PETR4", change_percent=3.0)],
        top_losers=[DigestMoverInput(symbol="VALE3", change_percent=-2.0)],
        watch_items=[
            DigestWatchItemInput(
                symbol="ITUB4",
                reason="concentration_above_threshold",
                detail="ITUB4 representa 40% da carteira.",
            )
        ],
        dividends_received=100.0,
    )
    defaults.update(overrides)
    return PortfolioDigestFactsInput(**defaults)


def test_prepare_prompt_includes_all_fact_fields():
    prompt = DigestNarrationService.prepare_prompt(build_facts())

    assert "PETR4" in prompt
    assert "VALE3" in prompt
    assert "ITUB4" in prompt
    assert "2026-08-11" in prompt
    assert "2026-08-18" in prompt
    assert "100.0" in prompt


def test_prepare_prompt_forbids_recommendation_language():
    prompt = DigestNarrationService.prepare_prompt(build_facts())

    assert "Nunca recomende" in prompt


def test_prepare_prompt_handles_empty_lists_without_crashing():
    facts = build_facts(top_gainers=[], top_losers=[], watch_items=[])

    prompt = DigestNarrationService.prepare_prompt(facts)

    assert "nenhum" in prompt
