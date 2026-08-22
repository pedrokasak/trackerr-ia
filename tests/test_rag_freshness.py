from dataclasses import dataclass
from datetime import date


from rag.freshness import assess_freshness, FRESHNESS_THRESHOLD_DAYS


@dataclass
class FakeChunk:
    source_type: str
    as_of: date | None


TODAY = date(2026, 8, 21)


def test_fresh_data_produces_no_note():
    chunks = [FakeChunk("portfolio_position", date(2026, 8, 20))]  # 1 dia, limite 2
    a = assess_freshness(chunks, TODAY)
    assert a.is_stale is False
    assert a.note == ""
    assert a.max_age_days == 1


def test_stale_position_annotates_the_answer():
    # posicao de 5 dias, limite 2 -> stale, mas nao hard (excesso 3 < 9)
    chunks = [FakeChunk("portfolio_position", date(2026, 8, 16))]
    a = assess_freshness(chunks, TODAY)
    assert a.is_stale is True
    assert a.is_hard_stale is False
    assert "5 dias" in a.note
    assert a.max_age_days == 5


def test_very_old_data_gets_a_strong_warning():
    # 30 dias, excesso enorme -> hard stale
    chunks = [FakeChunk("portfolio_position", date(2026, 7, 22))]
    a = assess_freshness(chunks, TODAY)
    assert a.is_hard_stale is True
    assert "⚠️" in a.note


def test_worst_chunk_dominates():
    # um fresco e um velho: o velho contamina a resposta
    chunks = [
        FakeChunk("portfolio_risk", date(2026, 8, 20)),  # fresco (limite 7)
        FakeChunk("portfolio_position", date(2026, 8, 14)),  # 7 dias, limite 2 -> stale
    ]
    a = assess_freshness(chunks, TODAY)
    assert a.is_stale is True


def test_risk_ages_slower_than_position():
    # mesmos 5 dias: posicao fica stale, risco nao
    pos = assess_freshness([FakeChunk("portfolio_position", date(2026, 8, 16))], TODAY)
    risk = assess_freshness([FakeChunk("portfolio_risk", date(2026, 8, 16))], TODAY)
    assert pos.is_stale is True
    assert risk.is_stale is False
    assert FRESHNESS_THRESHOLD_DAYS["portfolio_risk"] > FRESHNESS_THRESHOLD_DAYS["portfolio_position"]


def test_unknown_as_of_does_not_block_but_reports_none():
    chunks = [FakeChunk("portfolio_position", None)]
    a = assess_freshness(chunks, TODAY)
    assert a.max_age_days is None
    assert a.is_stale is False
    assert a.note == ""
