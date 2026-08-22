"""
Orquestra uma pergunta do RAG (TRA-37): embeda -> recupera contexto pessoal
(filtrado por user_id, nunca opcional) -> narra em cima do contexto
recuperado -> valida a resposta -> audita -> devolve.

Base fiscal generica (docs/rag-trackerr-ia.md secao 3.3) NAO esta
integrada aqui ainda: TRA-36 (curadoria de conteudo, revisao de contador)
segue bloqueada, e o schema atual de DocumentChunk exige user_id — conteudo
compartilhado entre todos os usuarios (a base fiscal) precisaria de uma
estrategia propria (sentinel de user_id ou tabela separada), que so faz
sentido desenhar quando houver conteudo real pra guardar.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from rag.audit import AuditLogRepository
from rag.embeddings import EmbeddingProvider
from rag.freshness import assess_freshness
from rag.repository import DocumentChunkRepository
from rag.response_guard import validate_rag_response
from benchmark.providers.base import LLMProvider

DISCLAIMER = (
    "Esta resposta é gerada automaticamente com base nos dados disponíveis "
    "e tem caráter educativo — não é recomendação de investimento nem "
    "cálculo definitivo de imposto. Consulte um profissional para decisões "
    "financeiras."
)

NO_CONTEXT_ANSWER = (
    "Não encontrei dados suficientes na sua carteira ou nos seus "
    "documentos pra responder essa pergunta com segurança."
)

GUARD_REJECTED_ANSWER = (
    "Não consigo responder com segurança dentro do que os dados "
    "confirmam agora. Tente reformular a pergunta."
)

TOP_K = 5


@dataclass
class RagQueryResult:
    answer: str
    source: str  # 'ai' | 'no_context' | 'guard_rejected'
    chunk_count: int
    data_max_age_days: int | None = None


class RagQueryService:
    def __init__(
        self,
        session: AsyncSession,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
    ) -> None:
        self._session = session
        self._embedding_provider = embedding_provider
        self._llm_provider = llm_provider
        self._chunks_repo = DocumentChunkRepository(session)
        self._audit_repo = AuditLogRepository(session)

    async def query(self, user_id: str, question: str) -> RagQueryResult:
        question = (question or "").strip()
        if not question:
            raise ValueError("question obrigatória.")

        query_embedding = await self._embedding_provider.embed(question)
        chunks = await self._chunks_repo.search(
            user_id=user_id, query_embedding=query_embedding, top_k=TOP_K
        )

        if not chunks:
            await self._audit_repo.record(
                user_id, question, [], NO_CONTEXT_ANSWER, "no_context"
            )
            return RagQueryResult(
                answer=NO_CONTEXT_ANSWER, source="no_context", chunk_count=0
            )

        prompt = self._build_prompt(question, chunks)
        raw = await self._llm_provider.analyze(prompt)
        raw_text = str(raw.get("answer") or raw.get("raw_response") or "")

        guard = validate_rag_response(raw_text)
        chunk_ids = [chunk.id for chunk in chunks]

        if not guard.valid:
            await self._audit_repo.record(
                user_id, question, chunk_ids, GUARD_REJECTED_ANSWER, guard.reason or "invalid"
            )
            return RagQueryResult(
                answer=GUARD_REJECTED_ANSWER,
                source="guard_rejected",
                chunk_count=len(chunks),
            )

        # Frescor em CODIGO, nao no prompt (TRA-77): dado velho apresentado
        # como atual e pior que ausencia de resposta. A nota, quando existe,
        # vai no TOPO — o usuario ve o aviso antes do conteudo.
        freshness = assess_freshness(chunks, datetime.now(timezone.utc).date())
        body = raw_text.strip()
        if freshness.note:
            body = f"{freshness.note}\n\n{body}"
        final_answer = f"{body}\n\n{DISCLAIMER}"
        await self._audit_repo.record(
            user_id,
            question,
            chunk_ids,
            final_answer,
            "ok",
            data_max_age_days=freshness.max_age_days,
        )
        return RagQueryResult(
            answer=final_answer,
            source="ai",
            chunk_count=len(chunks),
            data_max_age_days=freshness.max_age_days,
        )

    def _build_prompt(self, question: str, chunks: list) -> str:
        context_lines = "\n".join(f"- {chunk.content}" for chunk in chunks)
        return f"""
Você responde perguntas sobre a carteira e os documentos do usuário do Trakker, em português do Brasil.

REGRAS OBRIGATÓRIAS:
- Use APENAS o contexto abaixo. Não invente ativo, número ou dado que não esteja aqui.
- Nunca recomende compra, venda ou qualquer ação sobre um ativo. Descreva fatos, nunca instruções.
- Nunca afirme um valor de imposto como definitivo — trate qualquer cálculo fiscal como estimativa educativa.
- Se o contexto não for suficiente pra responder, diga isso explicitamente em vez de completar com suposição.

CONTEXTO:
{context_lines}

PERGUNTA:
{question}

Retorne APENAS JSON no formato:
{{"answer": "..."}}
"""
