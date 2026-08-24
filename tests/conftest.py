"""
Configuracao comum dos testes de endpoint.

Depois de TRA-89 todo endpoint que toca dado de usuario exige o header
`x-service-token`. Os testes de endpoint existem pra exercitar a REGRA de
cada rota, nao a autenticacao de servico — que tem cobertura propria em
`test_service_auth.py` (token certo, errado, ausente e nao configurado).

Por isso a dependencia e neutralizada aqui por padrao. A prova de que a
protecao esta de fato ligada nas rotas vive em
`test_endpoints_require_service_token.py`, que remove esse override.
"""

import pytest

from main import app
from rag.service_auth import require_service_token


async def _bypass_service_token() -> None:
    return None


@pytest.fixture(autouse=True)
def _allow_service_calls():
    app.dependency_overrides[require_service_token] = _bypass_service_token
    yield
    app.dependency_overrides.pop(require_service_token, None)
