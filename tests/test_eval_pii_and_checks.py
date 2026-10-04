"""PII e checagens determinísticas da avaliação offline (TRA-242)."""

import pytest

from evals.checks import answer_body, check_rag_answer
from evals.pii import contains_pii, scrub
from rag.query_service import DISCLAIMER


@pytest.mark.parametrize(
    "raw, placeholder",
    [
        ("meu cpf é 123.456.789-09", "[CPF]"),
        ("cpf 12345678909 cadastrado", "[CPF]"),
        ("cnpj 12.345.678/0001-90", "[CNPJ]"),
        ("me escreve em ana.silva+inv@gmail.com", "[EMAIL]"),
        ("liga no (11) 98765-4321", "[TELEFONE]"),
        ("whats +55 11 98765-4321", "[TELEFONE]"),
        ("cartão 4111 1111 1111 1111", "[CARTAO]"),
        ("agência 1234 conta 56789-0", "[CONTA]"),
    ],
)
def test_scrub_replaces_personal_data(raw, placeholder):
    cleaned = scrub(raw)

    assert placeholder in cleaned
    assert not contains_pii(cleaned)


def test_scrub_keeps_amounts_percentages_and_tickers():
    text = "PETR4 vale R$ 14.200,00 (23,1% da carteira); recebi R$ 3.284,50 em 12 meses."

    assert scrub(text) == text
    assert not contains_pii(text)


def test_answer_body_drops_disclaimer_and_freshness_note():
    answer = (
        "⚠️ Atenção: os dados da sua carteira usados nesta resposta têm cerca de 40 dias "
        "e podem estar desatualizados.\n\nVocê tem R$ 61.420,00.\n\n" + DISCLAIMER
    )

    assert answer_body(answer) == "Você tem R$ 61.420,00."


def test_flags_a_number_that_is_not_in_the_context():
    checks = check_rag_answer(
        "Quanto recebi?",
        "Proventos recebidos nos últimos 12 meses: R$ 3.284,50.",
        f"Você recebeu R$ 3.900,00.\n\n{DISCLAIMER}",
    )

    assert checks.unsupported_numbers == ["3.900,00"]
    assert checks.numeric_hallucination


def test_accepts_numbers_from_the_question_and_the_context():
    checks = check_rag_answer(
        "Se eu aportar R$ 1.000, como fica?",
        "Carteira total: R$ 61.420,00.",
        f"Com R$ 1.000 a mais, a carteira de R$ 61.420,00 cresce.\n\n{DISCLAIMER}",
    )

    assert checks.unsupported_numbers == []
    assert checks.disclaimer_present


def test_flags_recommendation_and_missing_disclaimer():
    checks = check_rag_answer("Devo vender?", "PETR4 subiu 19,0%.", "Recomendo vender PETR4.")

    assert checks.recommendation_language
    assert not checks.disclaimer_present
