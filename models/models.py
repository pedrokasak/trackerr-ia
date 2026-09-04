from datetime import date
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

# ============================================
# 1. MODELS
# ============================================

class StockMetrics(BaseModel):
    symbol: Optional[str] = ""
    roe_5y: Optional[float] = 0.0
    cagr_5y: Optional[float] = 0.0
    dividend_yield: Optional[float] = 0.0
    founding_year: Optional[int] = None
    is_leader: Optional[bool] = False
    sector_age: Optional[int] = 0
    is_blue_chip: Optional[bool] = False
    governance_score: Optional[float] = 0.0
    is_state_free: Optional[bool] = True
    net_debt_ebitda: Optional[float] = 0.0

class FiiMetrics(BaseModel):
    symbol: Optional[str] = ""
    property_age: Optional[int] = 0
    pvp_ratio: Optional[float] = 1.0
    dividend_years: Optional[int] = 0
    main_tenant_concentration: Optional[float] = 0.0
    main_property_concentration: Optional[float] = 0.0
    sector_yield_avg: Optional[float] = 0.0
    current_yield: Optional[float] = 0.0

class Asset(BaseModel):
    symbol: str
    type: str = "stock"
    quantity: Optional[float] = 0.0
    price: Optional[float] = 0.0
    current_price: Optional[float] = 0.0
    change_24h: Optional[float] = 0.0
    metrics: Optional[Dict[str, Any]] = Field(default_factory=dict)

class Portfolio(BaseModel):
    id: Optional[str] = "default"
    name: Optional[str] = "Principal"
    cpf: Optional[str] = ""
    assets: List[Asset] = Field(default_factory=list)
    total_value: Optional[float] = 0.0
    plan: Optional[str] = "free"

class UserProfile(BaseModel):
    user_id: str
    profile_plan: str = "free"
    portfolio: Portfolio = Field(default_factory=Portfolio)
    risk_profile: Optional[str] = "moderate"
    address: Optional[Dict[str, str]] = Field(default_factory=dict)
    preferences: Optional[Dict[str, str]] = Field(default_factory=dict)

# Outros modelos de resposta IA e simulação
class InvestmentScoreDetail(BaseModel):
    score: int = 0
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

class InvestmentScore(BaseModel):
    overall: int = 0
    diversification: int = 0
    risk: int = 0
    consistency: int = 0
    volatility: int = 0
    details: InvestmentScoreDetail = Field(default_factory=InvestmentScoreDetail)

class SimulationScenario(BaseModel):
    optimistic: float
    neutral: float
    pessimistic: float

class SimulationRequest(BaseModel):
    monthly_investment: float
    years: float
    current_portfolio_value: float
    expected_annual_return: Optional[float] = 0.10

class SimulationResponse(BaseModel):
    total_invested: float
    scenarios: SimulationScenario
    message: str

class ErrorDetection(BaseModel):
    type: str = "other"
    severity: str = "low"
    message: str
    symbol: Optional[str] = None

class OpportunityRadarItem(BaseModel):
    symbol: str
    type: str
    price: float
    rationale: str

class AllocationItem(BaseModel):
    category: str
    current: float
    ideal: float

class FeedItem(BaseModel):
    title: str
    content: str
    impact: str # positive, negative, neutral
    symbol: Optional[str] = None

class RebalancingResponse(BaseModel):
    ideal_allocation: List[AllocationItem]
    top_moves: List[str] # Sugestões de "reduzir X", "aumentar Y"

class AiAnalysisResult(BaseModel):
    portfolio_assessment: str
    investment_score: InvestmentScore
    error_detection: List[ErrorDetection] = Field(default_factory=list)
    opportunity_radar: List[OpportunityRadarItem] = Field(default_factory=list)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    risk_assessment: str
    rebalancing: Optional[RebalancingResponse] = None
    smart_feed: List[FeedItem] = Field(default_factory=list)


class ChatRequest(BaseModel):
    question: str
    profile_plan: Optional[str] = "free"
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str


# ============================================
# Digest semanal da carteira (TRA-17)
#
# O server (NestJS) decide todo fato — quais ativos, quais numeros, quais
# pontos de atencao. Este endpoint so recebe fatos ja fechados e devolve
# prosa. Nunca busca dado, nunca calcula, nunca escolhe o que citar.
# ============================================

class DigestMoverInput(BaseModel):
    symbol: str
    change_percent: float


class DigestWatchItemInput(BaseModel):
    symbol: str
    reason: str
    detail: str


class PortfolioDigestFactsInput(BaseModel):
    period_start: str
    period_end: str
    portfolio_value: Optional[float] = None
    period_change_pct: Optional[float] = None
    period_change_abs: Optional[float] = None
    top_gainers: List[DigestMoverInput] = Field(default_factory=list)
    top_losers: List[DigestMoverInput] = Field(default_factory=list)
    watch_items: List[DigestWatchItemInput] = Field(default_factory=list)
    dividends_received: Optional[float] = None


class DigestNarrateResponse(BaseModel):
    text: str


# ============================================
# Query RAG (TRA-37)
#
# user_id chega no corpo porque quem chama e o server (NestJS), ja
# autenticado — mesmo padrao do digest de e-mail. trackerr-ia nao faz
# auth propria, confia no chamador interno.
# ============================================

class RagQueryRequest(BaseModel):
    user_id: str
    question: str


class RagQueryResponse(BaseModel):
    answer: str
    source: str  # 'ai' | 'no_context' | 'guard_rejected'
    chunk_count: int
    # Frescor do pior chunk usado, em dias (TRA-77). None quando nenhum chunk
    # tinha data conhecida. Opcional: nao quebra quem ja consome a resposta.
    data_max_age_days: Optional[int] = None


# TRA-72: ingestao de fatos de carteira. NestJS decide o QUE virar chunk
# (posicao, radar de erro, etc.) e manda texto pronto — trackerr-ia so
# chunka, embeda e grava. Ver rag/ingestion_service.py.
class RagIngestItem(BaseModel):
    source_type: str
    source_id: str
    content: str
    # Metadata estruturada do fato (symbol, sector, portfolio_weight...) e
    # data de referencia. Opcionais pra nao quebrar quem ja chama o endpoint
    # com o payload de TRA-72.
    metadata: Optional[Dict[str, Any]] = None
    as_of: Optional[date] = None


class RagIngestRequest(BaseModel):
    user_id: str
    items: list[RagIngestItem]


class RagIngestResponse(BaseModel):
    chunks_deleted: int
    chunks_created: int
    chunks_unchanged: int = 0
    warnings: list[str] = Field(default_factory=list)


class RagEraseRequest(BaseModel):
    user_id: str


class RagEraseResponse(BaseModel):
    chunks_deleted: int
    # Linhas de auditoria preservadas com user_id anonimizado e texto
    # redigido, nao apagadas — ver rag/erasure_service.py pro porque.
    audit_rows_anonymized: int


# TRA-87: ingestao de conhecimento curado e compartilhado (base fiscal).
# Conteudo nao-pessoal, embedado uma vez, versionado. Endpoint administrativo.
class SharedKnowledgeItemModel(BaseModel):
    source_id: str
    content: str
    version: Optional[str] = None


class SharedKnowledgeIngestRequest(BaseModel):
    knowledge_base: str
    items: List[SharedKnowledgeItemModel]


class SharedKnowledgeIngestResponse(BaseModel):
    chunks_deleted: int
    chunks_created: int
    chunks_unchanged: int = 0
    warnings: List[str] = Field(default_factory=list)


# ============================================
# Insights com profundidade (TRA-56)
#
# Insight bruto ("Reduza exposicao a cripto") nao diz de onde veio nem o
# que fazer. As estruturas abaixo carregam a evidencia numerica que
# disparou o insight, uma confianca calculada a partir de metricas de
# entrada (nao pedida ao LLM), uma acao concreta com rota, e as fontes
# RAG consultadas. `title`/`body` permanecem para BC.
# ============================================


class InsightEvidence(BaseModel):
    """Ponto de dado deterministico que disparou o insight."""

    label: str
    value: Any
    source: Optional[str] = None  # id do fato de entrada, ex.: 'exposure.cripto'


class InsightConfidence(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    bucket: Literal["baixa", "media", "alta"]
    reason: str


class InsightAction(BaseModel):
    label: str
    route: str
    payload: Optional[Dict[str, Any]] = None
    why: Optional[str] = None


class InsightSource(BaseModel):
    """
    Reflexo do que TRA-76 ja emite quando um chunk RAG e usado. Campos sao
    opcionais pra acomodar tanto fatos pessoais (source_type/source_id) quanto
    conhecimento compartilhado (knowledge_base/source_id).
    """

    source_type: Optional[str] = None
    source_id: Optional[str] = None
    knowledge_base: Optional[str] = None
    as_of: Optional[date] = None


class Insight(BaseModel):
    id: str
    title: str
    body: str  # BC: campo curto ja consumido pelo frontend
    rationale: str
    evidence: List[InsightEvidence] = Field(default_factory=list)
    confidence: InsightConfidence
    action: Optional[InsightAction] = None
    sources: List[InsightSource] = Field(default_factory=list)


class InsightsRequest(BaseModel):
    user_profile: UserProfile
    data_freshness_days: Optional[int] = None


class InsightsResponse(BaseModel):
    insights: List[Insight]
