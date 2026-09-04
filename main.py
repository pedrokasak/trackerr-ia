from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.logger import logger as fastapi_logger
from datetime import datetime
from typing import Dict
import os
import logging
import uvicorn

from benchmark.benchmark import (
    AIAnalysisService,
    DigestNarrationService,
    FiiStrategy,
    StockStrategy,
    SimulationService,
)
from models.models import (
    FiiMetrics,
    StockMetrics,
    UserProfile,
    SimulationRequest,
    ChatRequest,
    ChatResponse,
    PortfolioDigestFactsInput,
    DigestNarrateResponse,
    RagQueryRequest,
    RagQueryResponse,
    RagIngestRequest,
    RagIngestResponse,
    RagEraseRequest,
    RagEraseResponse,
    SharedKnowledgeIngestRequest,
    SharedKnowledgeIngestResponse,
    InsightsRequest,
    InsightsResponse,
)
from insights.service import InsightsService
from insights.producers import PRODUCERS as LEGACY_INSIGHT_PRODUCERS
from benchmark.providers.factory import LLMFactory
from rag.database import get_rag_session
from rag.service_auth import require_service_token
from rag.embeddings import GeminiEmbeddingProvider
from rag.erasure_service import RagErasureService
from rag.shared_knowledge_service import (
    SharedKnowledgeService,
    SharedKnowledgeItem,
)
from rag.ingestion_service import RagIngestionService, RagIngestItem
from rag.query_service import RagQueryService

load_dotenv()

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

app = FastAPI(title="Hybrid Portfolio AI", version="2.5.0")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    fastapi_logger.error(f"Erro de validação: {exc.errors()}")
    fastapi_logger.error(f"Body: {await request.body()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(await request.body())},
    )

# ============================================
# ENDPOINTS
# ============================================

@app.post("/api/hybrid-analysis", dependencies=[Depends(require_service_token)])
async def hybrid_analysis(user_profile: UserProfile):
    """
    Análise Hybrid completa:
    - Free: Scores Básicos
    - Premium/Pro: AI (Groq/Llama) + Scores + Radar + Erros
    """
    try:
        fastapi_logger.info(f"Analisando portfolio {user_profile.user_id}")

        stock_analyses = {}
        fii_analyses = {}

        for asset in user_profile.portfolio.assets:
            if asset.type == "stock" and asset.metrics:
                # Usamos .evaluate() que é o novo nome no benchmark.py
                stock_analyses[asset.symbol] = StockStrategy.evaluate(asset.metrics)

            elif asset.type == "fii" and asset.metrics:
                fii_analyses[asset.symbol] = FiiStrategy.evaluate(asset.metrics)

        # 1. Se Free: retorna só análise estratégica
        if user_profile.profile_plan == "free":
            return {
                "schema_version": "v2",
                "plan": "free",
                "stock_scores": stock_analyses,
                "fii_scores": fii_analyses,
                "message": "Upgrade para Premium para análise com IA, Radar de Oportunidades e Detecção de Erros.",
                "timestamp": datetime.now().isoformat(),
            }

        # 2. Se Premium/Pro: IA faz análise completa (Score, Radar, Erros)
        prompt = AIAnalysisService.prepare_analysis_prompt(
            user_profile, stock_analyses, fii_analyses
        )

        ai_response = await AIAnalysisService.analyze_with_ai(prompt)

        # TRA-135: producers migrados para o shape estendido de Insight
        # (evidencia deterministica, confianca calculada, acao com rota,
        # rationale com guardrail anti-alucinacao). Rodam em paralelo ao
        # payload legado do LLM — consumidores antigos continuam lendo
        # `ai_analysis`, novos consumidores leem `insights_v2` chaveado por
        # `schema_version`.
        insights_service = InsightsService(
            llm_provider=LLMFactory.get_provider(),
            logger=fastapi_logger,
        )
        insights_v2: Dict[str, list] = {}
        for name, producer in LEGACY_INSIGHT_PRODUCERS.items():
            try:
                produced = await insights_service.generate(
                    user_profile, producer=producer
                )
                insights_v2[name] = [i.model_dump() for i in produced]
            except Exception as producer_error:  # pragma: no cover
                fastapi_logger.error(
                    f"Producer {name} falhou; seguindo com lista vazia: "
                    f"{producer_error}"
                )
                insights_v2[name] = []

        return {
            "schema_version": "v2",
            "plan": user_profile.user_id, # Usando ID para contexto
            "profile_plan": user_profile.profile_plan,
            "stock_scores": stock_analyses,
            "fii_scores": fii_analyses,
            "ai_analysis": ai_response, # Novo nome para o payload completo
            "insights_v2": insights_v2,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        fastapi_logger.error(f"Erro na análise: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate", dependencies=[Depends(require_service_token)])
async def simulate_portfolio(request: SimulationRequest):
    """
    Simulação de futuro baseada em aportes mensais
    """
    try:
        return SimulationService.simulate(request)
    except Exception as e:
        fastapi_logger.error(f"Erro na simulação: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", dependencies=[Depends(require_service_token)])
async def chat_portfolio(request: ChatRequest):
    """
    Chat inteligente baseado no contexto real da carteira.
    """
    try:
        result = await AIAnalysisService.chat_with_ai(
            question=request.question,
            profile_plan=request.profile_plan or "free",
            context=request.context or {},
        )
        answer = result.get("answer")
        if not answer:
            raw = result.get("raw_response")
            answer = raw if isinstance(raw, str) and raw.strip() else "Não consegui gerar resposta agora."
        return {"answer": str(answer)}
    except Exception as e:
        fastapi_logger.error(f"Erro no chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/api/portfolio-digest-narrate",
    response_model=DigestNarrateResponse,
    dependencies=[Depends(require_service_token)],
)
async def portfolio_digest_narrate(facts: PortfolioDigestFactsInput):
    """
    Narra os fatos do digest semanal de carteira (TRA-17). O NestJS manda
    fatos ja fechados e valida a resposta antes de usar — este endpoint so
    escreve prosa em cima do que recebeu.
    """
    try:
        text = await DigestNarrationService.narrate(facts)
        return {"text": text}
    except Exception as e:
        fastapi_logger.error(f"Erro ao narrar digest de carteira: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/rag/query",
    response_model=RagQueryResponse,
    dependencies=[Depends(require_service_token)],
)
async def rag_query(
    request: RagQueryRequest, session: AsyncSession = Depends(get_rag_session)
):
    """
    Pergunta em linguagem natural sobre a carteira/documentos do usuário
    (TRA-37). Retrieval sempre filtrado por user_id (rag/repository.py);
    resposta validada antes de sair (rag/response_guard.py); toda
    interação é auditada (rag/models.py:RagQueryAuditLog).
    """
    try:
        embedding_provider = GeminiEmbeddingProvider()
        service = RagQueryService(
            session=session,
            embedding_provider=embedding_provider,
            llm_provider=LLMFactory.get_provider(),
        )
        result = await service.query(request.user_id, request.question)
        return {
            "answer": result.answer,
            "source": result.source,
            "chunk_count": result.chunk_count,
            "data_max_age_days": result.data_max_age_days,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        fastapi_logger.error(f"Erro na query RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/rag/ingest",
    response_model=RagIngestResponse,
    dependencies=[Depends(require_service_token)],
)
async def rag_ingest(
    request: RagIngestRequest, session: AsyncSession = Depends(get_rag_session)
):
    """
    Recebe fatos ja prontos como texto (posicao, radar de erro, etc. —
    decisao de QUE virar chunk e do server, TRA-72) e sincroniza os chunks
    do usuario por diff de content_hash (TRA-74): so o que mudou paga
    embedding, o que sumiu da carteira e removido, o resto e pulado.
    `chunks_unchanged` na resposta mostra quanto a otimizacao economizou.
    """
    try:
        embedding_provider = GeminiEmbeddingProvider()
        service = RagIngestionService(session=session, embedding_provider=embedding_provider)
        items = [
            RagIngestItem(
                source_type=item.source_type,
                source_id=item.source_id,
                content=item.content,
                metadata=item.metadata,
                as_of=item.as_of,
            )
            for item in request.items
        ]
        result = await service.ingest(request.user_id, items)
        return {
            "chunks_deleted": result.chunks_deleted,
            "chunks_created": result.chunks_created,
            "chunks_unchanged": result.chunks_unchanged,
            "warnings": result.warnings,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        fastapi_logger.error(f"Erro na ingestão RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/rag/knowledge/ingest",
    response_model=SharedKnowledgeIngestResponse,
    dependencies=[Depends(require_service_token)],
)
async def rag_knowledge_ingest(
    request: SharedKnowledgeIngestRequest,
    session: AsyncSession = Depends(get_rag_session),
):
    """
    Ingestao de conhecimento CURADO e COMPARTILHADO (TRA-87), ex.: base fiscal
    revisada (TRA-36). Endpoint administrativo, rodado sob demanda quando uma
    nova revisao do conteudo e aprovada — NAO o cron diario por usuario.

    Conteudo compartilhado entre todos os usuarios, sem user_id: vive em
    tabela separada de document_chunks, sem tocar no isolamento por usuario.
    Diff incremental por content_hash: so re-embeda o que mudou.

    IMPORTANTE: so ingerir conteudo aprovado por profissional habilitado. O
    guardrail de resposta continua valendo, mas conteudo fiscal errado na
    base e responsabilidade de quem ingeriu.
    """
    try:
        embedding_provider = GeminiEmbeddingProvider()
        service = SharedKnowledgeService(
            session=session, embedding_provider=embedding_provider
        )
        items = [
            SharedKnowledgeItem(
                source_id=item.source_id,
                content=item.content,
                version=item.version,
            )
            for item in request.items
        ]
        result = await service.ingest(request.knowledge_base, items)
        return {
            "chunks_deleted": result.chunks_deleted,
            "chunks_created": result.chunks_created,
            "chunks_unchanged": result.chunks_unchanged,
            "warnings": result.warnings,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        fastapi_logger.error(f"Erro na ingestao de conhecimento compartilhado: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/rag/erase",
    response_model=RagEraseResponse,
    dependencies=[Depends(require_service_token)],
)
async def rag_erase(
    request: RagEraseRequest, session: AsyncSession = Depends(get_rag_session)
):
    """
    Apaga os dados de RAG de um usuario (TRA-78, LGPD).

    Endpoint explicito em vez de reaproveitar `/api/rag/ingest` com lista
    vazia: exclusao por direito do titular precisa aparecer como exclusao no
    log de acesso, nao disfarcada de sincronizacao de rotina.

    Idempotente — usuario sem dado nenhum devolve 200 com zeros. Isso
    importa pro chamador poder repetir com seguranca depois de um timeout.
    """
    try:
        service = RagErasureService(session=session)
        result = await service.erase(request.user_id)
        fastapi_logger.info(
            f"[LGPD] Dados de RAG apagados: chunks={result.chunks_deleted} "
            f"audit_anonimizados={result.audit_rows_anonymized}"
        )
        return {
            "chunks_deleted": result.chunks_deleted,
            "audit_rows_anonymized": result.audit_rows_anonymized,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        fastapi_logger.error(f"Erro na exclusao de dados RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/insights",
    response_model=InsightsResponse,
    dependencies=[Depends(require_service_token)],
)
async def generate_insights(request: InsightsRequest):
    """
    Gera insights com profundidade (TRA-56): evidencia deterministica,
    confianca calculada, acao com rota, rationale narrado com guardrail
    anti-alucinacao numerica. Ver `insights/service.py` para a divisao de
    trabalho entre codigo e LLM.
    """
    try:
        service = InsightsService(
            llm_provider=LLMFactory.get_provider(),
            logger=fastapi_logger,
        )
        insights = await service.generate(
            request.user_profile,
            data_freshness_days=request.data_freshness_days,
        )
        return {"insights": insights}
    except Exception as e:
        fastapi_logger.error(f"Erro ao gerar insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    # O provider vem da config, nao hardcoded: a versao anterior reportava
    # "Groq/Llama-3.3" fixo, que virou mentira quando o provider mudou (o
    # modelo nem existe mais). Endpoint de monitoramento que mente e pior
    # que endpoint que nao existe.
    return {
        "status": "ok",
        "version": "2.5.0",
        "llm_provider": os.getenv("LLM_PROVIDER", "gemini"),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
