"""
Prova que a autenticacao de servico esta LIGADA nas rotas (TRA-89).

O conftest neutraliza `require_service_token` no resto da suite pra que os
testes de endpoint exercitem a regra de cada rota. Aqui o override e
removido de proposito: sem isso, alguem poderia remover a dependencia de um
endpoint e a suite inteira continuaria verde.

Antes desta protecao, qualquer um com acesso de rede ao servico lia o
contexto financeiro de qualquer usuario informando o `user_id` no corpo,
apagava os dados de qualquer usuario, e envenenava a base de conhecimento
compartilhada.
"""

import pytest
from fastapi.testclient import TestClient

from main import app
from rag.service_auth import SERVICE_TOKEN_ENV, require_service_token

PROTECTED_ENDPOINTS = [
    ("/api/rag/query", {"user_id": "u1", "question": "quanto tenho?"}),
    ("/api/rag/ingest", {"user_id": "u1", "items": []}),
    ("/api/rag/erase", {"user_id": "u1"}),
    ("/api/rag/knowledge/ingest", {"knowledge_base": "fiscal", "items": []}),
    ("/api/chat", {"question": "oi"}),
    (
        "/api/insights",
        {"user_profile": {"user_id": "u1", "portfolio": {"assets": []}}},
    ),
]


@pytest.fixture
def client_without_bypass(monkeypatch):
    monkeypatch.setenv(SERVICE_TOKEN_ENV, "segredo-de-teste")
    app.dependency_overrides.pop(require_service_token, None)
    yield TestClient(app)


@pytest.mark.parametrize("path,payload", PROTECTED_ENDPOINTS)
def test_endpoint_recusa_requisicao_sem_token(client_without_bypass, path, payload):
    response = client_without_bypass.post(path, json=payload)

    assert response.status_code == 401, (
        f"{path} respondeu {response.status_code} sem token — a dependencia "
        "de autenticacao de servico saiu da rota."
    )


@pytest.mark.parametrize("path,payload", PROTECTED_ENDPOINTS)
def test_endpoint_recusa_token_errado(client_without_bypass, path, payload):
    response = client_without_bypass.post(
        path, json=payload, headers={"x-service-token": "chute"}
    )

    assert response.status_code == 401


def test_health_continua_publico(client_without_bypass):
    # Monitoramento precisa responder sem credencial, e o endpoint nao expoe
    # dado nenhum.
    response = client_without_bypass.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
