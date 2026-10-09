"""
Guardrail minimo do resumo de RI (TRA-238).

Diferente do guard do RAG pessoal (rag/response_guard.py), que barra
qualquer "venda"/"investir": texto de RI usa essas palavras como FATO
("venda da subsidiaria", "vai investir R$ 2 bi em capex"). Aqui so e
barrada a recomendacao DIRIGIDA ao leitor e o preco-alvo proprio. A
checagem de fidelidade numerica fica em TRA-239.
"""

import pytest

from rag.response_guard import validate_rag_response
from ri.fidelity import build_source_index
from ri.summary_guard import RawHighlight, enforce_fidelity, validate_ri_summary
from tests.ri_release_phrases import (
    DIRECTED_RECOMMENDATIONS,
    FACTUAL_RELEASE_PHRASES,
    PRICE_TARGETS,
)


# ------------------------------------------------ corpus de releases (TRA-239)


@pytest.mark.parametrize("phrase", FACTUAL_RELEASE_PHRASES)
def test_factual_release_phrase_passes(phrase):
    assert validate_ri_summary(highlights=[phrase], narrative="").valid is True


@pytest.mark.parametrize("phrase", DIRECTED_RECOMMENDATIONS)
def test_directed_recommendation_is_blocked(phrase):
    result = validate_ri_summary(highlights=[phrase], narrative="")
    assert (result.valid, result.reason) == (False, "directed_recommendation")


@pytest.mark.parametrize("phrase", PRICE_TARGETS)
def test_price_target_is_blocked(phrase):
    result = validate_ri_summary(highlights=[], narrative=phrase)
    assert (result.valid, result.reason) == (False, "price_target")


def test_personal_chat_guard_is_unchanged():
    # O guard do RAG pessoal continua barrando "venda" em qualquer contexto:
    # e por isso que o RI tem guard proprio, e nao uma flexibilizacao dele.
    assert validate_rag_response("Conclusão da venda da subsidiária.").valid is False
    assert validate_rag_response("A carteira está concentrada em bancos.").valid is True


# ------------------------------------------------ fidelidade ao documento


SOURCE_TEXT = (
    "Destaques do 2T26. A receita líquida atingiu R$ 12,3 bilhões, 8,1% "
    "superior ao 2T25. -- 1 of 2 -- O EBITDA ajustado somou R$ 4,2 bilhões, "
    "com margem de 34,1%. A Companhia concluiu a venda da subsidiária de gás. "
    "-- 2 of 2 --"
)


def enforce(highlights, narrative=""):
    return enforce_fidelity(
        highlights=highlights,
        narrative=narrative,
        source=build_source_index(SOURCE_TEXT),
        metadata=["PETR4", "Petrobras", "2T26", "2026-08-07"],
    )


def test_keeps_highlight_backed_by_a_real_excerpt_and_cites_its_page():
    result = enforce(
        [
            RawHighlight(
                text="EBITDA ajustado de R$ 4,2 bilhões, com margem de 34,1%.",
                evidence="O EBITDA ajustado somou R$ 4,2 bilhões, com margem de 34,1%",
            )
        ]
    )

    assert result.dropped == []
    [cited] = result.highlights
    assert cited.page == 2
    assert cited.excerpt.startswith("O EBITDA ajustado somou R$ 4,2 bilhões")


def test_drops_highlight_whose_excerpt_is_not_in_the_document():
    result = enforce(
        [
            RawHighlight(
                text="Receita de R$ 12,3 bilhões.",
                evidence="A receita da Companhia foi recorde, de R$ 12,3 bilhões",
            )
        ]
    )

    assert result.highlights == []
    assert result.dropped == ["highlight_evidence_not_found"]


def test_drops_highlight_with_number_missing_from_its_excerpt():
    # O modelo arredondou 8,1% para 8%: o numero nao esta no trecho citado.
    result = enforce(
        [
            RawHighlight(
                text="Receita cresceu 8% no ano.",
                evidence="A receita líquida atingiu R$ 12,3 bilhões, 8,1% superior ao 2T25",
            )
        ]
    )

    assert result.highlights == []
    assert result.dropped == ["highlight_number_not_in_evidence"]


def test_drops_highlight_citing_a_period_that_does_not_exist_in_the_document():
    result = enforce(
        [
            RawHighlight(
                text="Receita de R$ 12,3 bilhões, acima do 1T26.",
                evidence="A receita líquida atingiu R$ 12,3 bilhões, 8,1% superior ao 2T25",
            )
        ]
    )

    assert result.dropped == ["highlight_unknown_identifier"]


def test_accepts_period_and_ticker_from_document_metadata():
    result = enforce(
        [
            RawHighlight(
                text="PETR4: receita de R$ 12,3 bilhões no 2T26.",
                evidence="A receita líquida atingiu R$ 12,3 bilhões, 8,1% superior ao 2T25",
            )
        ]
    )

    assert result.dropped == []


def test_drops_narrative_with_a_number_absent_from_the_document():
    result = enforce([], narrative="O lucro líquido cresceu 15% no trimestre.")

    assert result.narrative == ""
    assert result.dropped == ["narrative_unsupported_claim"]


def test_keeps_narrative_whose_numbers_are_in_the_document():
    narrative = "Trimestre de receita de R$ 12,3 bilhões e margem EBITDA de 34,1%."

    assert enforce([], narrative=narrative).narrative == narrative


def test_excerpt_shown_to_the_user_has_no_page_marker():
    result = enforce(
        [
            RawHighlight(
                text="Receita 8,1% acima do 2T25 e EBITDA de R$ 4,2 bilhões.",
                evidence="8,1% superior ao 2T25. O EBITDA ajustado somou R$ 4,2 bilhões",
            )
        ]
    )

    [cited] = result.highlights
    assert "of 2" not in cited.excerpt
    assert cited.page == 1


def test_accepts_factual_corporate_language():
    result = validate_ri_summary(
        highlights=[
            "Receita líquida cresceu 12% no trimestre.",
            "Conclusão da venda da subsidiária de gás.",
            "A companhia vai investir R$ 2 bilhões em capex em 2027.",
        ],
        narrative="O conselho aprovou a recomendação de pagamento de JCP.",
    )
    assert result.valid is True


def test_accepts_sentence_starting_with_factual_sale():
    result = validate_ri_summary(
        highlights=["Venda já concluída da participação na distribuidora."],
        narrative="",
    )
    assert result.valid is True


def test_rejects_imperative_buy_or_sell():
    result = validate_ri_summary(
        highlights=["Compre a ação antes do próximo balanço."],
        narrative="",
    )
    assert result.valid is False
    assert result.reason == "directed_recommendation"


def test_rejects_second_person_recommendation():
    result = validate_ri_summary(
        highlights=[],
        narrative="Com esse resultado, você deveria vender suas ações.",
    )
    assert result.valid is False
    assert result.reason == "directed_recommendation"


def test_rejects_we_recommend_to_reader():
    result = validate_ri_summary(
        highlights=[],
        narrative="Recomendamos a compra do papel.",
    )
    assert result.valid is False
    assert result.reason == "directed_recommendation"


def test_rejects_own_price_target():
    result = validate_ri_summary(
        highlights=["Preço-alvo de R$ 45,00 para os próximos 12 meses."],
        narrative="",
    )
    assert result.valid is False
    assert result.reason == "price_target"


def test_rejects_empty_output():
    result = validate_ri_summary(highlights=[], narrative="   ")
    assert result.valid is False
    assert result.reason == "empty"
