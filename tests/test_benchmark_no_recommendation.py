# -*- coding: utf-8 -*-
"""
TRA-53: as estrategias de benchmark emitiam COMPRA/HOLD/VENDA por ativo.

Isso e indicacao de investimento, e o produto declara publicamente que nao
faz isso ("O Trackerr nao e consultoria de investimento e nao indica
ativos", FAQ da landing). Nao ha registro de consultoria.

O `score` e o `rating` continuam: descrevem qualidade observada a partir de
regras explicitas. O que saiu foi o salto de "score 68" para "VENDA".
"""

import pytest

from benchmark.benchmark import StockStrategy, FiiStrategy


ACAO_OTIMA = {
    "roe": 25,
    "cagr_5y": 15,
    "dividend_yield": 8,
    "net_debt_ebitda": 1,
    "governance_score": 90,
}
ACAO_RUIM = {
    "roe": 2,
    "cagr_5y": 1,
    "dividend_yield": 0,
    "net_debt_ebitda": 9,
    "governance_score": 10,
}
FII_OTIMO = {
    "pvp": 0.9,
    "dividend_yield": 12,
    "vacancy_rate": 2,
    "main_tenant_concentration": 10,
    "dividend_years": 10,
}
FII_RUIM = {
    "pvp": 2.0,
    "dividend_yield": 1,
    "vacancy_rate": 40,
    "main_tenant_concentration": 90,
    "dividend_years": 0,
}


@pytest.mark.parametrize("metrics", [ACAO_OTIMA, ACAO_RUIM])
def test_acao_nao_emite_recomendacao(metrics):
    resultado = StockStrategy.evaluate(metrics)
    assert "recommendation" not in resultado
    # O que descreve qualidade observada permanece.
    assert "score" in resultado
    assert "rating" in resultado


@pytest.mark.parametrize("metrics", [FII_OTIMO, FII_RUIM])
def test_fii_nao_emite_recomendacao(metrics):
    resultado = FiiStrategy.evaluate(metrics)
    assert "recommendation" not in resultado
    assert "score" in resultado
    assert "rating" in resultado


@pytest.mark.parametrize(
    "metrics,strategy",
    [
        (ACAO_OTIMA, StockStrategy),
        (ACAO_RUIM, StockStrategy),
        (FII_OTIMO, FiiStrategy),
        (FII_RUIM, FiiStrategy),
    ],
)
def test_nenhum_verbo_de_ordem_no_resultado(metrics, strategy):
    """Nem em campo proprio, nem escondido dentro de `details`."""
    texto = str(strategy.evaluate(metrics)).upper()
    for verbo in ("COMPRA", "VENDA", "HOLD"):
        assert verbo not in texto
