from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag.erasure_service import (
    ANONYMIZED_USER_ID,
    REDACTED_TEXT,
    RagErasureService,
)
from rag.repository import MissingUserIdError


def make_session(audit_rowcount=0):
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(rowcount=audit_rowcount))
    session.commit = AsyncMock()
    return session


@patch("rag.erasure_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_apaga_chunks_e_anonimiza_auditoria(mock_repo_cls):
    mock_repo_cls.return_value.delete_for_user = AsyncMock(return_value=7)
    session = make_session(audit_rowcount=3)

    result = await RagErasureService(session).erase("user-1")

    mock_repo_cls.return_value.delete_for_user.assert_awaited_once_with("user-1")
    assert result.chunks_deleted == 7
    assert result.audit_rows_anonymized == 3


@patch("rag.erasure_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_auditoria_e_redigida_nao_apagada(mock_repo_cls):
    """
    O ponto da decisao de TRA-78: a linha de auditoria SOBREVIVE (valor de
    seguranca, principio 4.3), mas sem o conteudo pessoal. Se algum dia
    alguem trocar isto por um DELETE, este teste quebra.
    """
    mock_repo_cls.return_value.delete_for_user = AsyncMock(return_value=0)
    session = make_session()

    await RagErasureService(session).erase("user-1")

    statement = session.execute.call_args.args[0]
    compiled = str(statement)
    assert compiled.strip().upper().startswith("UPDATE"), compiled
    values = statement.compile().params
    assert values["user_id"] == ANONYMIZED_USER_ID
    assert values["question"] == REDACTED_TEXT
    assert values["response_text"] == REDACTED_TEXT


@patch("rag.erasure_service.DocumentChunkRepository")
@pytest.mark.asyncio
async def test_e_idempotente_para_usuario_sem_dado(mock_repo_cls):
    # Chamador precisa poder repetir com seguranca depois de um timeout.
    mock_repo_cls.return_value.delete_for_user = AsyncMock(return_value=0)
    session = make_session(audit_rowcount=0)

    result = await RagErasureService(session).erase("user-sem-dado")

    assert result.chunks_deleted == 0
    assert result.audit_rows_anonymized == 0


@pytest.mark.asyncio
async def test_rejeita_user_id_vazio():
    with pytest.raises(MissingUserIdError):
        await RagErasureService(MagicMock()).erase("")
