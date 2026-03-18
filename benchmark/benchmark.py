import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi.logger import logger

# Mocking Prophet to avoid dependency issues if not installed
try:
    from prophet import Prophet as prophet
except ImportError:
    class prophet:
        def __init__(self, **kwargs): pass
        def fit(self, df): pass
        def make_future_dataframe(self, periods): return pd.DataFrame({"ds": pd.date_range(start=datetime.now(), periods=periods)})
        def predict(self, df): 
            df["yhat"] = 100 + np.random.randn(len(df))
            df["yhat_lower"] = df["yhat"] - 2
            df["yhat_upper"] = df["yhat"] + 2
            return df

from models.models import UserProfile, Asset, StockMetrics, FiiMetrics, SimulationRequest
from .providers.factory import LLMFactory

class StockStrategy:
    """Estratégia para Ações"""
    @staticmethod
    def evaluate(metrics: Dict[str, Any]) -> Dict[str, Any]:
        score = 0
        details = []
        
        # Filtros simplificados
        if metrics.get("roe_5y", 0) > 15:
            score += 20
            details.append("ROE > 15%")
        if metrics.get("cagr_5y", 0) > 10:
            score += 20
            details.append("Crescimento Receita > 10%")
        if metrics.get("dividend_yield", 0) > 5:
            score += 15
            details.append("DY > 5%")
        if metrics.get("net_debt_ebitda", 99) < 2:
            score += 20
            details.append("Dívida/EBITDA < 2x")
        if metrics.get("governance_score", 0) > 80:
            score += 25
            details.append("Alta governança")

        rating = "TOP" if score >= 80 else "BOM" if score >= 50 else "EVITAR"
        recommendation = "COMPRA" if score >= 70 else "HOLD" if score >= 40 else "VENDA"
        
        return {
            "score": score,
            "rating": rating,
            "recommendation": recommendation,
            "details": details
        }

class FiiStrategy:
    """Estratégia para FIIs"""
    @staticmethod
    def evaluate(metrics: Dict[str, Any]) -> Dict[str, Any]:
        score = 0
        details = []
        
        pvp = metrics.get("pvp_ratio", 2.0)
        yield_curr = metrics.get("current_yield", 0)
        
        if pvp < 1.1:
            score += 30
            details.append("P/VP atrativo")
        elif pvp > 1.5:
            score -= 20
            details.append("P/VP muito alto")

        if yield_curr > metrics.get("sector_yield_avg", 0):
            score += 30
            details.append("Yield acima da média do setor")

        if metrics.get("main_tenant_concentration", 100) < 25:
            score += 20
            details.append("Baixa concentração de inquilino")

        if metrics.get("dividend_years", 0) > 5:
            score += 20
            details.append("Histórico sólido de dividendos")

        rating = "TOP" if score >= 75 else "BOM" if score >= 50 else "ALERTA"
        recommendation = "COMPRA" if score >= 70 else "HOLD" if score >= 40 else "VENDA"
        
        return {
            "score": score,
            "rating": rating,
            "recommendation": recommendation,
            "details": details,
            "critical_rejection": pvp > 1.5
        }

class SimulationService:
    @staticmethod
    def simulate(req: SimulationRequest) -> Dict[str, Any]:
        total_invested = req.current_portfolio_value + (req.monthly_investment * req.years * 12)
        
        # Compound interest logic
        r_neutral = req.expected_annual_return
        r_optimistic = r_neutral + 0.05
        r_pessimistic = r_neutral - 0.05
        
        months = req.years * 12
        
        def future_value(pv, pmt, r, n):
            if r == 0: return pv + (pmt * n)
            rm = (1 + r)**(1/12) - 1
            fv = pv * (1 + rm)**n + pmt * (((1 + rm)**n - 1) / rm)
            return fv

        neutral_fv = future_value(req.current_portfolio_value, req.monthly_investment, r_neutral, months)
        optimistic_fv = future_value(req.current_portfolio_value, req.monthly_investment, r_optimistic, months)
        pessimistic_fv = future_value(req.current_portfolio_value, req.monthly_investment, r_pessimistic, months)

        return {
            "total_invested": float(total_invested),
            "scenarios": {
                "optimistic": float(optimistic_fv),
                "neutral": float(neutral_fv),
                "pessimistic": float(pessimistic_fv)
            },
            "message": f"Em {req.years} anos, seu patrimônio pode chegar a R$ {neutral_fv:,.2f}."
        }

class PredictiveService:
    @staticmethod
    def forecast_price(symbol: str) -> Dict[str, Any]:
        """
        Gera uma previsão simplificada usando Prophet (ou Mock)
        """
        try:
            # Dados fake para o Prophet (ds, y)
            df = pd.DataFrame({
                "ds": pd.date_range(start="2024-01-01", periods=100),
                "y": np.random.randint(10, 100, 100)
            })
            
            m = prophet()
            m.fit(df)
            
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)
            
            return {
                "symbol": symbol,
                "ds": forecast["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "yhat": forecast["yhat"].tolist(),
                "yhat_lower": forecast["yhat_lower"].tolist(),
                "yhat_upper": forecast["yhat_upper"].tolist(),
            }
        except Exception as e:
            logger.error(f"Erro no forecast via PredictiveService: {e}")
            return {}

class AIAnalysisService:
    @staticmethod
    def prepare_analysis_prompt(
        user_profile: UserProfile,
        stock_analyses: Dict[str, Dict] = None,
        fii_analyses: Dict[str, Dict] = None,
        forecasts: Dict[str, Dict] = None,
    ) -> str:
        prompt = f"""
            # Análise Profissional de Portfolio Trakker (Nível Multi-Family Office)
            Perfil: {user_profile.risk_profile} | Plano: {user_profile.profile_plan}
            Total Investido: R$ {user_profile.portfolio.total_value:,.2f}

            ## CARTEIRA ATUAL:
            """
        assets_by_category = {"renda_fixa": 0, "acoes": 0, "fii": 0, "etf": 0, "cripto": 0, "outros": 0}
        total_val = user_profile.portfolio.total_value or 1
        
        for asset in user_profile.portfolio.assets:
            val = asset.quantity * asset.current_price
            prompt += f"\n- {asset.symbol}: {asset.quantity} un @ R${asset.current_price:.2f} (Total: R${val:.2f}) - {asset.change_24h}% hoje"
            
            cat = asset.type
            if cat in assets_by_category:
                assets_by_category[cat] += val
            else:
                assets_by_category["outros"] += val

        prompt += f"""
## TAREFAS OBRIGATÓRIAS (RETORNE APENAS JSON):
1. **Investment Score (0-100)**: Avalie diversificação, risco, consistência, volatilidade.
2. **Auto Rebalanceamento**: Sugira uma "Carteira Ideal" baseada no perfil {user_profile.risk_profile}.
3. **Smart Feed (Spotify Style)**: Gere 3-5 eventos recentes impactantes (ex: 'Petrobras caiu 2%', 'Cripto subiu 5%') e o impacto total na carteira.
4. **Radar de Oportunidades**: 3 recomendações claras.
5. **Radar Anti-Erro**: Alertas críticos.

## JSON SCHEMA:
{{
    "portfolio_assessment": "...",
    "investment_score": {{ "overall": 0, "diversification": 0, "risk": 0, "consistency": 0, "volatility": 0, "details": {{ "score": 0, "strengths": [], "weaknesses": [], "recommendations": [] }} }},
    "rebalancing": {{
        "ideal_allocation": [
            {{ "category": "Renda Fixa", "current": {(assets_by_category['renda_fixa']/total_val)*100}, "ideal": 40 }},
            {{ "category": "Ações", "current": {(assets_by_category['acoes']/total_val)*100}, "ideal": 30 }},
            {{ "category": "FIIs", "current": {(assets_by_category['fii']/total_val)*100}, "ideal": 15 }},
            {{ "category": "Cripto", "current": {(assets_by_category['cripto']/total_val)*100}, "ideal": 5 }},
            {{ "category": "ETFs", "current": {(assets_by_category['etf']/total_val)*100}, "ideal": 10 }}
        ],
        "top_moves": ["Reduzir Ações (70% -> 30%)", "Aumentar Renda Fixa"]
    }},
    "smart_feed": [
        {{ "title": "Impacto do Dia", "content": "Seu portfólio oscilou +1.2% hoje puxado por Bitcoin.", "impact": "positive", "symbol": "BTC" }},
        {{ "title": "Petrobras (PETR4)", "content": "Caiu 2.1% devido a rumores de dividendos.", "impact": "negative", "symbol": "PETR4" }}
    ],
    "error_detection": [ {{ "type": "correlation|concentration", "severity": "high|medium|low", "message": "...", "symbol": "..." }} ],
    "opportunity_radar": [ {{ "symbol": "...", "type": "...", "price": 0, "target_price": 0, "upside": 0.0, "rationale": "..." }} ],
    "risk_assessment": "..."
}}
"""
        return prompt

    @staticmethod
    async def analyze_with_ai(prompt: str) -> Dict[str, Any]:
        provider = LLMFactory.get_provider()
        return await provider.analyze(prompt)
