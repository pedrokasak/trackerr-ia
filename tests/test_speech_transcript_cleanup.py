"""Correção de ticker mal ouvido e frase inventada (ADR-007, spike de 12/11/2026)."""

import pytest

from speech.transcript_cleanup import clean_transcript, normalize_tickers, strip_courtesy_prefix

CARTEIRA = ["PETR4", "VALE3", "HGLG11", "ITUB4", "IVVB11"]


@pytest.mark.parametrize(
    "ouvido, esperado",
    [
        # Casos reais do spike com gravações de celular.
        ("Quanto a Petri4 pagou de dividendos esse ano?", "Quanto a PETR4 pagou de dividendos esse ano?"),
        ("A Peter 4 pagou R$ 1.248,30", "A PETR4 pagou R$ 1.248,30"),
        ("Se eu vender 200 ações do YouTube 4 a R$ 32,50", "Se eu vender 200 ações do ITUB4 a R$ 32,50"),
        ("A Vale 3 subiu ou caiu esse mês?", "A VALE3 subiu ou caiu esse mês?"),
        ("Qual o peso do HGLG 11 na minha carteira?", "Qual o peso do HGLG11 na minha carteira?"),
        ("Compara o IVVB11 com o CDI", "Compara o IVVB11 com o CDI"),
        ("Quanto vai cair do Itaú 4?", "Quanto vai cair do ITUB4?"),
    ],
)
def test_corrige_ticker_mal_ouvido(ouvido, esperado):
    assert normalize_tickers(ouvido, CARTEIRA) == esperado


@pytest.mark.parametrize(
    "texto",
    [
        "Quanto recebo até o dia 15 de dezembro?",
        "Compara o IVVB11 com o CDI dos últimos 12 meses.",
        "Vendi 200 ações em 2024",
    ],
)
def test_nao_mexe_em_numero_que_nao_e_ticker(texto):
    assert normalize_tickers(texto, CARTEIRA) == texto


def test_sem_carteira_devolve_o_texto_igual():
    assert normalize_tickers("A Petri4 pagou", []) == "A Petri4 pagou"


def test_empate_entre_dois_tickers_nao_corrige():
    # Melhor não corrigir do que escolher o ticker errado.
    assert normalize_tickers("A ABCD 3 subiu", ["ABCE3", "ABCF3"]) == "A ABCD 3 subiu"


def test_remove_frase_de_cortesia_inventada_no_inicio():
    assert strip_courtesy_prefix("Obrigado.  A Vale 3 subiu ou caiu esse mês?") == "A Vale 3 subiu ou caiu esse mês?"


def test_mantem_obrigado_que_faz_parte_da_pergunta():
    texto = "Obrigado pela resposta, e a PETR4?"
    assert strip_courtesy_prefix(texto) == texto


def test_limpeza_completa():
    assert clean_transcript(" Obrigado. A Vale 3 subiu? ", CARTEIRA) == "A VALE3 subiu?"


def test_obrigado_sozinho_vira_texto_vazio():
    # O transcriber já devolve o texto sem espaço no fim.
    assert clean_transcript("Obrigado.", CARTEIRA) == ""


@pytest.mark.parametrize(
    "texto",
    [
        "Quanto vale 3 cotas do HGLG11?",
        "Quanto vale 4 mil reais em PETR4?",
        "Comprei vale 3 ações ontem",
    ],
)
def test_numero_seguido_de_quantidade_nao_vira_ticker(texto):
    assert normalize_tickers(texto, CARTEIRA) == texto


def test_palavra_curta_exige_semelhanca_maior():
    # "mês" e MRVE têm semelhança ~0,57: não pode virar MRVE3.
    assert normalize_tickers("No mês 3 recebi", ["MRVE3"]) == "No mês 3 recebi"
