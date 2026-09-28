"""
Fidelidade do resumo de RI ao documento-fonte (TRA-239).

Funcoes puras: nenhuma chamada de LLM. Elas decidem se um numero, um codigo
de periodo ou um trecho citado pelo modelo EXISTE no texto que ele recebeu.
"""

from decimal import Decimal

from ri.fidelity import (
    build_source_index,
    extract_identifiers,
    locate_excerpt,
    number_values,
    page_at,
    unsupported_identifiers,
    unsupported_numbers,
)


def values(text: str) -> set:
    return {value for _, candidates in number_values(text) for value in candidates}


# ---------------------------------------------------------------- numeros


def test_reads_brazilian_number_format():
    assert values("receita de R$ 1.234,56 milhões") == {Decimal("1234.56")}
    assert values("alta de 12,3%") == {Decimal("12.3")}


def test_reads_english_number_format_from_english_releases():
    assert Decimal("1234.5") in values("net revenue of US$ 1,234.5 million")


def test_ambiguous_thousands_keep_both_readings():
    # "1.234" e mil duzentos e trinta e quatro em pt-BR e 1,234 em en-US.
    assert values("1.234") == {Decimal("1234"), Decimal("1.234")}


def test_equivalent_spellings_match():
    assert not unsupported_numbers("margem de 2,0%", frozenset(values("margem de 2%")))


def test_ignores_digits_glued_to_letters():
    # Codigo de periodo, ticker e ordinal nao sao quantidades.
    assert values("no 2T26 a PETR4 ficou em 3º lugar") == set()


def test_keeps_numbers_followed_by_unit_suffixes():
    assert values("R$ 2,5bi e alavancagem de 1,8x") == {Decimal("2.5"), Decimal("1.8")}


def test_flags_number_absent_from_reference():
    reference = frozenset(values("A receita líquida cresceu 8,1% no trimestre."))
    assert unsupported_numbers("Receita cresceu 8%", reference) == ["8"]
    assert unsupported_numbers("Receita cresceu 8,1%", reference) == []


def test_ignores_page_markers_as_numbers():
    index = build_source_index("Receita de R$ 10 bi. -- 7 of 40 -- Lucro de R$ 2 bi.")
    assert Decimal("7") not in index.numbers
    assert Decimal("40") not in index.numbers
    assert {Decimal("10"), Decimal("2")} <= index.numbers


# ---------------------------------------------------------- identificadores


def test_extracts_period_codes_and_tickers():
    assert extract_identifiers("No 2T26 a PETR4 superou o 2Q25 e o 1S2026") == {
        "2T26",
        "PETR4",
        "2Q25",
        "1S26",
    }


def test_flags_period_code_absent_from_reference():
    allowed = frozenset({"2T26", "2T25"})
    assert unsupported_identifiers("Alta frente ao 1T26", allowed) == ["1T26"]
    assert unsupported_identifiers("Alta frente ao 2T25", allowed) == []


# ------------------------------------------------------------------ trechos


SOURCE = (
    "Destaques do trimestre. A receita líquida atingiu R$ 12,3 bilhões, "
    "8,1% superior ao 2T25. -- 1 of 3 -- "
    "O EBITDA ajus-  tado somou R$ 4,2 bilhões, com margem de 34,1%. "
    "-- 2 of 3 -- A dívida líquida encerrou o período em R$ 9,8 bilhões."
)


def test_locates_verbatim_excerpt():
    index = build_source_index(SOURCE)
    position = locate_excerpt(index, "A receita líquida atingiu R$ 12,3 bilhões")
    assert position is not None
    assert SOURCE[position:].startswith("A receita")


def test_locates_excerpt_despite_pdf_spacing_hyphenation_and_accents():
    index = build_source_index(SOURCE)
    # O PDF quebrou "ajustado" com hifen e espaco; o modelo reescreveu certo.
    assert locate_excerpt(index, "O EBITDA ajustado somou R$ 4,2 bilhoes") is not None


def test_does_not_locate_paraphrased_excerpt():
    index = build_source_index(SOURCE)
    assert locate_excerpt(index, "A receita da companhia foi de R$ 12,3 bilhões") is None


def test_rejects_excerpt_too_short_to_prove_anything():
    index = build_source_index(SOURCE)
    assert locate_excerpt(index, "receita") is None


def test_locates_excerpt_across_page_break():
    index = build_source_index(SOURCE)
    assert locate_excerpt(index, "8,1% superior ao 2T25. O EBITDA ajustado somou") is not None


def test_page_comes_from_the_marker_that_closes_the_page():
    index = build_source_index(SOURCE)
    assert page_at(index, locate_excerpt(index, "A receita líquida atingiu R$ 12,3")) == 1
    assert page_at(index, locate_excerpt(index, "O EBITDA ajustado somou R$ 4,2")) == 2
    assert page_at(index, locate_excerpt(index, "A dívida líquida encerrou o período")) == 3


def test_page_is_unknown_without_markers():
    index = build_source_index("A receita líquida atingiu R$ 12,3 bilhões no trimestre.")
    assert page_at(index, locate_excerpt(index, "A receita líquida atingiu R$ 12,3")) is None
