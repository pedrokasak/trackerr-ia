"""
Exclusao dos dados de RAG de um usuario (TRA-78, LGPD).

O RAG replica dado financeiro pessoal num Postgres separado do banco
transacional. Sem trilha de exclusao, apagar a conta no server deixava
`document_chunks` e `rag_query_audit_log` intactos indefinidamente — o que
e exposicao de LGPD, nao divida tecnica.

## Decisao sobre o audit log (explicita, nao default acidental)

As duas tabelas recebem tratamento diferente porque tem naturezas
diferentes:

- `document_chunks` e cache derivado. Zero valor de auditoria, todo o
  conteudo e dado pessoal. **Apaga a linha.**
- `rag_query_audit_log` existe pro principio 4.3 do CLAUDE.md (acao critica
  tem que ser auditavel), mas guarda a PERGUNTA e a RESPOSTA — texto livre
  sobre a carteira do usuario, que e dado pessoal e frequentemente
  identificavel por si so. Anonimizar so o `user_id` nao resolveria: a
  pergunta continuaria la.

  Entao: **preserva a linha, anonimiza o `user_id` e redige o texto.** O que
  sobrevive e o fato auditavel (houve consulta, nesta data, com este
  veredito do guard) sem o conteudo pessoal. Contagem de consultas e taxa
  de bloqueio do guard continuam calculaveis; o que o usuario perguntou,
  nao.

Apagar a linha inteira perderia o sinal de seguranca; manter intacta
violaria o direito de exclusao. Redigir preserva os dois lados.
"""

from dataclasses import dataclass

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from rag.models import RagQueryAuditLog
from rag.repository import DocumentChunkRepository, MissingUserIdError

# Sentinel no lugar do user_id nas linhas de auditoria preservadas. Nao e um
# user_id valido de propósito: qualquer query por usuario real nunca casa.
ANONYMIZED_USER_ID = "__erased__"

REDACTED_TEXT = "[removido por exclusao de conta]"


@dataclass
class RagErasureResult:
    chunks_deleted: int
    audit_rows_anonymized: int


class RagErasureService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._chunks_repo = DocumentChunkRepository(session)

    async def erase(self, user_id: str) -> RagErasureResult:
        if not user_id:
            raise MissingUserIdError(
                "user_id obrigatorio — exclusao nunca roda sem escopo de usuario."
            )

        chunks_deleted = await self._chunks_repo.delete_for_user(user_id)

        result = await self._session.execute(
            update(RagQueryAuditLog)
            .where(RagQueryAuditLog.user_id == user_id)
            .values(
                user_id=ANONYMIZED_USER_ID,
                question=REDACTED_TEXT,
                response_text=REDACTED_TEXT,
            )
        )
        await self._session.commit()

        return RagErasureResult(
            chunks_deleted=chunks_deleted,
            audit_rows_anonymized=result.rowcount or 0,
        )
