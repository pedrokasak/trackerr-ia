import pytest
from fastapi import HTTPException

from rag.service_auth import SERVICE_TOKEN_ENV, require_service_token


@pytest.mark.asyncio
async def test_aceita_token_correto(monkeypatch):
    monkeypatch.setenv(SERVICE_TOKEN_ENV, "segredo-do-deploy")

    assert await require_service_token("segredo-do-deploy") is None


@pytest.mark.asyncio
async def test_recusa_token_errado(monkeypatch):
    monkeypatch.setenv(SERVICE_TOKEN_ENV, "segredo-do-deploy")

    with pytest.raises(HTTPException) as exc:
        await require_service_token("chute")
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_recusa_requisicao_sem_header(monkeypatch):
    monkeypatch.setenv(SERVICE_TOKEN_ENV, "segredo-do-deploy")

    with pytest.raises(HTTPException) as exc:
        await require_service_token(None)
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_falha_fechado_quando_segredo_nao_configurado(monkeypatch):
    # Sem segredo o servico recusa TUDO. Aceitar aqui transformaria um erro
    # de deploy numa porta aberta silenciosa.
    monkeypatch.delenv(SERVICE_TOKEN_ENV, raising=False)

    with pytest.raises(HTTPException) as exc:
        await require_service_token("qualquer-coisa")
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_segredo_vazio_conta_como_nao_configurado(monkeypatch):
    monkeypatch.setenv(SERVICE_TOKEN_ENV, "   ")

    with pytest.raises(HTTPException) as exc:
        await require_service_token("   ")
    assert exc.value.status_code == 503
