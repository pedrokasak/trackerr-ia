"""
Testes do repositorio do vector store. Nao dependem de Postgres real —
inspecionam a query compilada (SQLAlchemy permite isso sem executar nada),
o que basta pra travar o invariante que importa: toda busca filtra por
user_id, sempre.

Cobertura contra Postgres/pgvector de verdade fica pendente (Docker
indisponivel nesta sessao) — rodar as migracoes e uma busca real antes de
confiar em producao.
"""

import pytest

from rag.repository import DocumentChunkRepository, MissingUserIdError


class TestBuildSearchStatement:
    def test_rejeita_user_id_vazio(self):
        with pytest.raises(MissingUserIdError):
            DocumentChunkRepository.build_search_statement(
                user_id="", query_embedding=[0.1] * 768
            )

    def test_rejeita_user_id_none(self):
        with pytest.raises(MissingUserIdError):
            DocumentChunkRepository.build_search_statement(
                user_id=None, query_embedding=[0.1] * 768  # type: ignore[arg-type]
            )

    def test_rejeita_top_k_nao_positivo(self):
        with pytest.raises(ValueError):
            DocumentChunkRepository.build_search_statement(
                user_id="user-1", query_embedding=[0.1] * 768, top_k=0
            )

    def test_query_sempre_filtra_por_user_id(self):
        """
        O teste que importa de verdade (risco P0 do doc): a clausula WHERE
        da query compilada precisa conter o filtro de user_id — nao um
        filtro condicional, sempre presente na string SQL gerada.
        """
        statement = DocumentChunkRepository.build_search_statement(
            user_id="user-123", query_embedding=[0.1] * 768
        )
        compiled = str(statement.compile(compile_kwargs={"literal_binds": False}))

        assert "document_chunks.user_id" in compiled
        assert "WHERE" in compiled

    def test_source_type_e_filtro_adicional_nao_substituto(self):
        """source_type nunca pode aparecer sozinho — sempre em conjunto
        com o filtro de user_id, nunca no lugar dele."""
        statement = DocumentChunkRepository.build_search_statement(
            user_id="user-123",
            query_embedding=[0.1] * 768,
            source_type="brokerage_note",
        )
        compiled = str(statement.compile(compile_kwargs={"literal_binds": False}))

        assert "document_chunks.user_id" in compiled
        assert "document_chunks.source_type" in compiled

    def test_top_k_vira_limit_na_query(self):
        statement = DocumentChunkRepository.build_search_statement(
            user_id="user-123", query_embedding=[0.1] * 768, top_k=3
        )
        compiled = str(statement.compile(compile_kwargs={"literal_binds": True}))

        assert "LIMIT 3" in compiled


class TestDeleteForUser:
    @pytest.mark.asyncio
    async def test_rejeita_user_id_vazio(self):
        repo = DocumentChunkRepository(session=None)  # type: ignore[arg-type]
        with pytest.raises(MissingUserIdError):
            await repo.delete_for_user("")
