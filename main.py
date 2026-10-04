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
    RiSummaryRequest,
    RiSummaryResponse,
    RiIndexRequest,
    RiIndexResponse,
    RiAskRequest,
    RiAskResponse,
    ChatPlanRequest,
    ChatPlanResponse,
)
from benchmark.providers.base import ToolSpec
from chat.tool_planner import SingleStepToolRuntime
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
from ri.summary_service import RiSummaryRejectedError, RiSummaryService
from ri.knowledge_service import (
    RiAnswerRejectedError,
    RiDocumentAnswerer,
    RiDocumentIndexer,
    RiDocumentMeta,
)

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
    "/api/ri/summarize",
    response_model=RiSummaryResponse,
    dependencies=[Depends(require_service_token)],
)
async def ri_summarize(request: RiSummaryRequest):
    """
    Resume um documento de RI (TRA-238). O server manda o texto ja extraido
    e os sinais calculados por regra, e guarda o resultado em cache pela
    hash do conteudo — cada documento e resumido uma vez, nao uma vez por
    usuario.

    Cada destaque volta com a citacao que o sustenta (TRA-239); o que o
    documento nao sustenta e descartado e contado em `dropped_claims`.

    422 quando a saida do modelo e inutilizavel ou barrada pelo guardrail:
    o server trata qualquer nao-2xx como falha e cai no resumo estruturado.
    """
    try:
        result = await RiSummaryService.summarize(request)
        if result.dropped_reasons:
            # So os motivos: o texto do resumo nao vai pro log.
            fastapi_logger.warning(
                f"Resumo de RI ({request.document.ticker}) descartou "
                f"{len(result.dropped_reasons)} afirmacao(oes): "
                f"{', '.join(sorted(set(result.dropped_reasons)))}"
            )
        return {
            "highlights": result.highlights,
            "narrative": result.narrative,
            "provider": result.provider,
            "citations": [
                {"highlight": item.text, "excerpt": item.excerpt, "page": item.page}
                for item in result.citations
            ],
            "dropped_claims": len(result.dropped_reasons),
        }
    except RiSummaryRejectedError as e:
        fastapi_logger.warning(
            f"Resumo de RI rejeitado ({request.document.ticker}): {e.reason}"
        )
        raise HTTPException(status_code=422, detail=e.reason)
    except Exception as e:
        fastapi_logger.error(f"Erro ao resumir documento de RI: {e}")
        raise HTTPException(status_code=500, detail="ri_summary_failed")


@app.post(
    "/api/ri/index",
    response_model=RiIndexResponse,
    dependencies=[Depends(require_service_token)],
)
async def ri_index(
    request: RiIndexRequest, session: AsyncSession = Depends(get_rag_session)
):
    """
    Guarda o texto de um documento de RI no acervo (TRA-264), em chunks por
    pagina. O server manda o texto que ja extraiu; mesmo documento com o
    mesmo texto nao paga embedding de novo.
    """
    try:
        document = request.document
        indexer = RiDocumentIndexer(
            session=session, embedding_provider=GeminiEmbeddingProvider()
        )
        result = await indexer.index(
            RiDocumentMeta(
                key=document.key,
                issuer=document.issuer,
                ticker=document.ticker,
                company=document.company,
                title=document.title,
                category=document.category,
                document_type=document.document_type,
                period=document.period,
                published_at=document.published_at,
                source_url=document.source_url,
            ),
            request.content,
        )
        return {"status": result.status, "chunks": result.chunks}
    except Exception as e:
        fastapi_logger.error(f"Erro ao indexar documento de RI: {e}")
        raise HTTPException(status_code=500, detail="ri_index_failed")


@app.post(
    "/api/ri/ask",
    response_model=RiAskResponse,
    dependencies=[Depends(require_service_token)],
)
async def ri_ask(
    request: RiAskRequest, session: AsyncSession = Depends(get_rag_session)
):
    """
    Responde uma pergunta sobre os documentos de RI de UM emissor (TRA-264).
    Cada afirmacao volta com documento, pagina e o trecho real que a
    sustenta; sem trecho que sustente, `not_found`.

    422 quando a saida do modelo e barrada pelo guardrail (recomendacao,
    preco-alvo): o server responde que nao conseguiu, sem texto do modelo.
    """
    try:
        answerer = RiDocumentAnswerer(
            session=session,
            embedding_provider=GeminiEmbeddingProvider(),
            llm_provider=LLMFactory.get_provider(),
        )
        result = await answerer.ask(
            request.issuer, request.question, request.published_after
        )
        if result.dropped_reasons:
            # So os motivos: o texto das afirmacoes nao vai pro log.
            fastapi_logger.warning(
                f"Resposta do acervo de RI ({request.issuer}) descartou "
                f"{len(result.dropped_reasons)} afirmacao(oes): "
                f"{', '.join(sorted(set(result.dropped_reasons)))}"
            )
        return {
            "answer": [
                {
                    "text": item.text,
                    "citation": {
                        "document_key": item.chunk.document_key,
                        "title": item.chunk.title,
                        "category": item.chunk.category,
                        "period": item.chunk.period,
                        "published_at": item.chunk.published_at,
                        "source_url": item.chunk.source_url,
                        "page": item.page,
                        "excerpt": item.excerpt,
                    },
                }
                for item in result.items
            ],
            "not_found": result.not_found,
            "provider": result.provider,
            "dropped_claims": len(result.dropped_reasons),
        }
    except RiAnswerRejectedError as e:
        fastapi_logger.warning(f"Resposta do acervo de RI rejeitada ({request.issuer}): {e.reason}")
        raise HTTPException(status_code=422, detail=e.reason)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        fastapi_logger.error(f"Erro ao responder pelo acervo de RI: {e}")
        raise HTTPException(status_code=500, detail="ri_ask_failed")


@app.post(
    "/api/chat/plan",
    response_model=ChatPlanResponse,
    dependencies=[Depends(require_service_token)],
)
async def chat_plan(request: ChatPlanRequest):
    """
    Roteador do chat com tool-calling (TRA-241). Escolhe quais intenções
    determinísticas do server respondem à pergunta: de 1 a 3 chamadas, com
    os tickers. Não executa nada nem responde à pergunta; o server executa e
    monta a resposta, sem número vindo do modelo.

    Sem provider com tool calling na cadeia: 200 com `reason: not_supported`,
    e o server responde pela rota de antes.
    """
    try:
        runtime = SingleStepToolRuntime(LLMFactory.get_provider())
        plan = await runtime.plan(
            request.question,
            [
                ToolSpec(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                )
                for tool in request.tools
            ],
            request.max_calls,
        )
        # Só nomes e contagens no log: a pergunta é do usuário.
        fastapi_logger.info(
            f"Roteador do chat: {len(plan.calls)} chamada(s) "
            f"[{', '.join(call.name for call in plan.calls)}] "
            f"via {plan.provider or '-'} ({plan.reason or 'ok'})"
        )
        return {
            "calls": [
                {"name": call.name, "arguments": call.arguments}
                for call in plan.calls
            ],
            "provider": plan.provider,
            "input_tokens": plan.input_tokens,
            "output_tokens": plan.output_tokens,
            "reason": plan.reason,
        }
    except Exception as e:
        fastapi_logger.error(f"Erro no roteador do chat: {type(e).__name__}")
        raise HTTPException(status_code=502, detail="chat_plan_failed")


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
