"""
Autenticacao de servico-a-servico do trackerr-ia (TRA-89).

O trackerr-ia nao faz auth de usuario final: quem autentica a pessoa e o
server (NestJS), e por isso o `user_id` chega no corpo da requisicao. O
problema e que ate agora NADA impedia um terceiro de mandar a mesma
requisicao com o `user_id` de outra pessoa — `/api/rag/query` devolvia o
contexto financeiro de qualquer usuario informado, `/api/rag/erase`
destruia dados, e `/api/rag/knowledge/ingest` envenenava a base de
conhecimento compartilhada (a mesma que vai receber o conteudo fiscal).
A unica protecao era isolamento de rede, que some no instante em que o
servico ganha URL publica.

O segredo compartilhado NAO substitui isolamento de rede — soma. Continua
valendo nao publicar este servico na internet.

Comparacao com `secrets.compare_digest` de proposito: `==` em string vaza
o tamanho do prefixo correto pelo tempo de execucao.
"""

import os
import secrets

from fastapi import Header, HTTPException, status

SERVICE_TOKEN_ENV = "TRACKERR_IA_SERVICE_TOKEN"
SERVICE_TOKEN_HEADER = "x-service-token"


class ServiceTokenNotConfiguredError(RuntimeError):
    """O servico nao pode subir sem segredo configurado."""


def _expected_token() -> str:
    token = (os.getenv(SERVICE_TOKEN_ENV) or "").strip()
    if not token:
        # Falha fechado. Aceitar requisicao quando o segredo nao esta
        # configurado transformaria um erro de deploy em porta aberta
        # silenciosa — exatamente o modo de falha que este modulo existe
        # pra impedir.
        raise ServiceTokenNotConfiguredError(
            f"{SERVICE_TOKEN_ENV} nao configurada — o trackerr-ia recusa "
            "requisicoes ate o segredo compartilhado com o server existir."
        )
    return token


async def require_service_token(
    x_service_token: str | None = Header(default=None),
) -> None:
    """
    Dependencia FastAPI: exige o segredo compartilhado com o server.

    Usada nos endpoints que leem, gravam ou apagam dado de usuario. O
    `/api/health` fica de fora de proposito — monitoramento precisa
    responder sem credencial, e ele nao expoe dado nenhum.
    """
    try:
        expected = _expected_token()
    except ServiceTokenNotConfiguredError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        )

    provided = (x_service_token or "").strip()
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de servico invalido ou ausente.",
        )
