from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from main import app


client = TestClient(app)

VALID_PAYLOAD = {
    "period_start": "2026-08-11",
    "period_end": "2026-08-18",
    "portfolio_value": 10000.0,
    "period_change_pct": 5.0,
    "period_change_abs": 500.0,
    "top_gainers": [{"symbol": "PETR4", "change_percent": 3.0}],
    "top_losers": [{"symbol": "VALE3", "change_percent": -2.0}],
    "watch_items": [
        {
            "symbol": "ITUB4",
            "reason": "concentration_above_threshold",
            "detail": "ITUB4 representa 40% da carteira.",
        }
    ],
    "dividends_received": 100.0,
}


def test_portfolio_digest_narrate_returns_text():
    with patch(
        "benchmark.benchmark.DigestNarrationService.narrate",
        new=AsyncMock(return_value="Sua carteira subiu 5% na semana, puxada por PETR4."),
    ):
        response = client.post("/api/portfolio-digest-narrate", json=VALID_PAYLOAD)

    assert response.status_code == 200
    assert response.json() == {
        "text": "Sua carteira subiu 5% na semana, puxada por PETR4."
    }


def test_portfolio_digest_narrate_accepts_minimal_facts():
    minimal_payload = {
        "period_start": "2026-08-11",
        "period_end": "2026-08-18",
    }
    with patch(
        "benchmark.benchmark.DigestNarrationService.narrate",
        new=AsyncMock(return_value="Sem posições no período."),
    ):
        response = client.post(
            "/api/portfolio-digest-narrate", json=minimal_payload
        )

    assert response.status_code == 200


def test_portfolio_digest_narrate_returns_500_on_provider_error():
    with patch(
        "benchmark.benchmark.DigestNarrationService.narrate",
        new=AsyncMock(side_effect=Exception("provider timeout")),
    ):
        response = client.post("/api/portfolio-digest-narrate", json=VALID_PAYLOAD)

    assert response.status_code == 500
