from rag.response_guard import validate_rag_response


def test_aceita_resposta_factual_sem_recomendacao():
    result = validate_rag_response(
        "Sua carteira tem PETR4 representando 15% do total, com yield de 8% no período."
    )
    assert result.valid is True


def test_rejeita_resposta_vazia():
    assert validate_rag_response("").valid is False
    assert validate_rag_response("   ").valid is False


def test_rejeita_linguagem_de_recomendacao():
    result = validate_rag_response("Recomendo vender PETR4 agora.")
    assert result.valid is False
    assert result.reason == "recommendation_language"


def test_rejeita_afirmacao_definitiva_de_imposto():
    result = validate_rag_response("Você deve pagar R$ 1.200 de imposto sobre esse ganho.")
    assert result.valid is False
    assert result.reason == "definitive_tax_claim"


def test_nao_confunde_investir_generico_com_recomendacao_explicita():
    # "investir" ainda cai no deny-list — teste documenta o comportamento
    # atual (conservador: prefere falso positivo a falso negativo aqui).
    result = validate_rag_response("Historicamente investir em ações renderia mais.")
    assert result.valid is False
    assert result.reason == "recommendation_language"


# TRA-242: a avaliação offline achou o guardrail barrando a resposta que
# recusava recomendar — e o usuário recebia "não consigo responder" no lugar
# dos fatos.
def test_aceita_a_recusa_explicita_de_recomendar():
    result = validate_rag_response(
        "Não posso fazer recomendações de compra, venda ou qualquer ação sobre "
        "seus ativos. Você possui 200 ações de PETR4 com preço médio de R$ 32,10."
    )
    assert result.valid is True


def test_continua_barrando_ordem_negativa():
    result = validate_rag_response("Não venda PETR4 agora.")
    assert result.valid is False
    assert result.reason == "recommendation_language"


def test_recusa_seguida_de_recomendacao_continua_barrada():
    result = validate_rag_response(
        "Não posso fazer recomendações, mas recomendo vender PETR4."
    )
    assert result.valid is False
    assert result.reason == "recommendation_language"


def test_recomendacao_fora_da_recusa_continua_barrada():
    result = validate_rag_response(
        "Não posso dar sugestões personalizadas. Compre mais VALE3."
    )
    assert result.valid is False


def test_recusa_com_recomendacao_na_mesma_frase_continua_barrada():
    for text in (
        "Não posso fazer recomendações, compre PETR4 se quiser.",
        "Não posso dar sugestões de compra, invista em VALE3.",
        "Não posso fazer recomendações de venda agora, venda tudo.",
    ):
        result = validate_rag_response(text)
        assert result.valid is False, text
        assert result.reason == "recommendation_language"


def test_aceita_recusas_comuns_do_modelo():
    for text in (
        "Não posso fazer recomendações de investimento. Sua carteira tem 6 ativos.",
        "Não consigo dar indicações personalizadas sobre seus ativos.",
        "Não é possível fazer sugestões de compra ou venda. PETR4 subiu 19,0%.",
    ):
        assert validate_rag_response(text).valid is True, text
