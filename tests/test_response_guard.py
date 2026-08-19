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
