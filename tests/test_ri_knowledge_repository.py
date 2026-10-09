"""
Repositorio do acervo de RI (TRA-264). Como em test_rag_repository.py, a
query compilada e inspecionada sem Postgres: o invariante que importa e que
toda busca filtra por emissor.
"""

from datetime import date

import pytest

from ri.knowledge_repository import MissingIssuerError, RiDocumentChunkRepository


def compiled(statement) -> str:
    return str(statement.compile(compile_kwargs={"literal_binds": False}))


def test_toda_busca_filtra_por_emissor():
    sql = compiled(
        RiDocumentChunkRepository.build_search_statement("petr", [0.1] * 768, 8)
    )

    assert "ri_document_chunks.issuer" in sql
    assert "WHERE" in sql
    assert "LIMIT" in sql


@pytest.mark.parametrize("issuer", ["", "   ", None])
def test_recusa_busca_sem_emissor(issuer):
    with pytest.raises(MissingIssuerError):
        RiDocumentChunkRepository.build_search_statement(issuer, [0.1] * 768, 8)


def test_recusa_top_k_nao_positivo():
    with pytest.raises(ValueError):
        RiDocumentChunkRepository.build_search_statement("PETR", [0.1] * 768, 0)


def test_janela_de_datas_e_filtro_adicional_nunca_substituto():
    sql = compiled(
        RiDocumentChunkRepository.build_search_statement(
            "PETR", [0.1] * 768, 8, published_after=date(2026, 1, 1)
        )
    )

    assert "ri_document_chunks.issuer" in sql
    assert "ri_document_chunks.published_at >=" in sql


def test_emissor_normalizado_em_maiusculas():
    statement = RiDocumentChunkRepository.build_search_statement(" petr ", [0.1] * 768, 8)
    params = statement.compile().params

    assert "PETR" in params.values()
