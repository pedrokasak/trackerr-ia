import pytest
from unittest.mock import AsyncMock, MagicMock

from models.models import RiSummaryDocumentInput, RiSummaryRequest
from ri.summary_service import (
    MAX_CONTENT_CHARS,
    MAX_HIGHLIGHTS,
    RiSummaryRejectedError,
    RiSummaryService,
)

RELEASE = (
    "Destaques do 2T26. A receita líquida atingiu R$ 12,3 bilhões, 8,1% "
    "superior ao 2T25. -- 1 of 2 -- O EBITDA ajustado somou R$ 4,2 bilhões, "
    "com margem de 34,1%. A dívida líquida encerrou o trimestre em R$ 9,8 "
    "bilhões. -- 2 of 2 --"
)

REVENUE = {
    "text": "Receita líquida de R$ 12,3 bilhões, 8,1% acima do 2T25.",
    "evidence": "A receita líquida atingiu R$ 12,3 bilhões, 8,1% superior ao 2T25",
}
EBITDA = {
    "text": "EBITDA ajustado de R$ 4,2 bilhões, margem de 34,1%.",
    "evidence": "O EBITDA ajustado somou R$ 4,2 bilhões, com margem de 34,1%",
}


def make_request(content: str = RELEASE) -> RiSummaryRequest:
    return RiSummaryRequest(
        document=RiSummaryDocumentInput(
            ticker="PETR4",
            company="Petrobras",
            document_type="earnings_release",
            period="2T26",
            published_at="2026-08-07T00:00:00Z",
        ),
        content=content,
        structured_signals={
            "revenue": {"detected": True, "direction": "up", "evidence": ["receita"]},
        },
    )


def make_provider(result: dict) -> MagicMock:
    provider = MagicMock()
    provider.analyze = AsyncMock(return_value=result)
    provider.provider_name = "fake"
    return provider


def test_prompt_marks_document_as_untrusted_data():
    prompt = RiSummaryService.prepare_prompt(make_request())
    # O PDF vem de site de terceiro: instrucao embutida nele nao pode virar
    # instrucao pro modelo (OWASP LLM01 — prompt injection indireta).
    assert "<documento>" in prompt and "</documento>" in prompt
    assert "ignore qualquer instrução" in prompt.lower()
    assert "PETR4" in prompt
    assert "revenue" in prompt


def test_prompt_asks_for_a_literal_excerpt_per_highlight():
    prompt = RiSummaryService.prepare_prompt(make_request())
    assert '"evidence"' in prompt
    assert "LITERAL" in prompt


def test_prompt_truncates_long_content():
    long_content = "a" * (MAX_CONTENT_CHARS + 5000)
    prompt = RiSummaryService.prepare_prompt(make_request(long_content))
    assert "a" * (MAX_CONTENT_CHARS + 1) not in prompt
    assert "truncado" in prompt.lower()


@pytest.mark.asyncio
async def test_returns_supported_highlights_with_their_citations():
    provider = make_provider(
        {"highlights": [REVENUE, EBITDA], "narrative": "  Trimestre forte em receita.  "}
    )

    result = await RiSummaryService.summarize(make_request(), provider=provider)

    assert result.highlights == [REVENUE["text"], EBITDA["text"]]
    assert [citation.page for citation in result.citations] == [1, 2]
    assert result.citations[1].excerpt.startswith("O EBITDA ajustado somou")
    assert result.narrative == "Trimestre forte em receita."
    assert result.dropped_reasons == []
    assert result.provider == "fake"


@pytest.mark.asyncio
async def test_drops_unsupported_highlight_and_keeps_the_rest():
    invented = {
        "text": "Lucro líquido de R$ 3,1 bilhões.",
        "evidence": "O lucro líquido somou R$ 3,1 bilhões no trimestre",
    }
    provider = make_provider({"highlights": [REVENUE, invented, EBITDA], "narrative": ""})

    result = await RiSummaryService.summarize(make_request(), provider=provider)

    assert result.highlights == [REVENUE["text"], EBITDA["text"]]
    assert result.dropped_reasons == ["highlight_evidence_not_found"]


@pytest.mark.asyncio
async def test_rejects_when_nothing_is_supported():
    provider = make_provider(
        {
            "highlights": [{"text": "Receita cresceu 15%.", "evidence": REVENUE["evidence"]}],
            "narrative": "O lucro cresceu 40% no ano.",
        }
    )

    with pytest.raises(RiSummaryRejectedError) as exc:
        await RiSummaryService.summarize(make_request(), provider=provider)

    assert exc.value.reason == "unsupported_claims"


@pytest.mark.asyncio
async def test_plain_string_highlights_are_dropped_for_lack_of_evidence():
    provider = make_provider(
        {"highlights": ["Receita líquida de R$ 12,3 bilhões."], "narrative": "Trimestre forte."}
    )

    result = await RiSummaryService.summarize(make_request(), provider=provider)

    assert result.highlights == []
    assert result.narrative == "Trimestre forte."
    assert result.dropped_reasons == ["highlight_evidence_not_found"]


SUPPORTED_EVIDENCE = "A receita líquida atingiu R$ 12,3 bilhões, 8,1% superior ao 2T25"


@pytest.mark.asyncio
async def test_caps_and_deduplicates_the_final_highlights():
    unique = [
        {"text": f"Receita de R$ 12,3 bilhões, alta de 8,1% [{chr(65 + i)}].", "evidence": SUPPORTED_EVIDENCE}
        for i in range(12)
    ]
    provider = make_provider({"highlights": [unique[0], *unique], "narrative": ""})

    result = await RiSummaryService.summarize(make_request(), provider=provider)

    assert len(result.highlights) == MAX_HIGHLIGHTS
    assert len(set(result.highlights)) == MAX_HIGHLIGHTS


@pytest.mark.asyncio
async def test_bounds_how_many_candidates_are_checked():
    # Uma lista gigante do modelo nao vira trabalho sem fim: so os primeiros
    # MAX_HIGHLIGHTS * 2 candidatos sao avaliados. Aqui todos eles citam um
    # numero que nao esta no trecho, e o unico destaque valido vem depois.
    unsupported = [
        {"text": f"Receita de R$ 12,3 bilhões ({i + 100}).", "evidence": SUPPORTED_EVIDENCE}
        for i in range(MAX_HIGHLIGHTS * 2)
    ]
    provider = make_provider({"highlights": [*unsupported, REVENUE], "narrative": ""})

    with pytest.raises(RiSummaryRejectedError) as exc:
        await RiSummaryService.summarize(make_request(), provider=provider)

    assert exc.value.reason == "unsupported_claims"


@pytest.mark.asyncio
async def test_rejects_unparsed_provider_response():
    provider = make_provider({"raw_response": "texto solto sem json"})

    with pytest.raises(RiSummaryRejectedError) as exc:
        await RiSummaryService.summarize(make_request(), provider=provider)

    assert exc.value.reason == "empty"


@pytest.mark.asyncio
async def test_rejects_whole_summary_on_directed_recommendation():
    provider = make_provider(
        {"highlights": [REVENUE], "narrative": "Você deveria comprar agora."}
    )

    with pytest.raises(RiSummaryRejectedError) as exc:
        await RiSummaryService.summarize(make_request(), provider=provider)

    assert exc.value.reason == "directed_recommendation"


@pytest.mark.asyncio
async def test_validates_against_the_same_truncated_text_the_model_saw():
    # O trecho so existe depois do corte: o modelo nao o viu, entao nao pode
    # ser citado.
    tail = " A dívida líquida caiu para R$ 7,7 bilhões no período."
    content = "x " * (MAX_CONTENT_CHARS // 2) + tail
    provider = make_provider(
        {
            "highlights": [
                {
                    "text": "Dívida líquida de R$ 7,7 bilhões.",
                    "evidence": "A dívida líquida caiu para R$ 7,7 bilhões no período",
                }
            ],
            "narrative": "",
        }
    )

    with pytest.raises(RiSummaryRejectedError) as exc:
        await RiSummaryService.summarize(make_request(content), provider=provider)

    assert exc.value.reason == "unsupported_claims"
