"""
Guardrail minimo do resumo de RI (TRA-238).

Diferente do guard do RAG pessoal (rag/response_guard.py), que barra
qualquer "venda"/"investir": texto de RI usa essas palavras como FATO
("venda da subsidiaria", "vai investir R$ 2 bi em capex"). Aqui so e
barrada a recomendacao DIRIGIDA ao leitor e o preco-alvo proprio. A
checagem de fidelidade numerica fica em TRA-239.
"""

from ri.summary_guard import validate_ri_summary


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
