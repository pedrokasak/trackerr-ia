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
    target_price: float
    upside: float
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
