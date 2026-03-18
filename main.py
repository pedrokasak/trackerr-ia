from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.logger import logger as fastapi_logger
from datetime import datetime
import logging
import uvicorn

from benchmark.benchmark import (
    AIAnalysisService,
    FiiStrategy,
    PredictiveService,
    StockStrategy,
    SimulationService,
)
from models.models import FiiMetrics, StockMetrics, UserProfile, SimulationRequest

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

@app.post("/api/hybrid-analysis")
async def hybrid_analysis(user_profile: UserProfile):
    """
    Análise Hybrid completa:
    - Free: Prophet só + Scores Básicos
    - Premium/Pro: Prophet + AI (Groq/Llama) + Scores + Radar + Erros
    """
    try:
        fastapi_logger.info(f"Analisando portfolio {user_profile.user_id}")

        stock_analyses = {}
        fii_analyses = {}
        forecasts = {}

        for asset in user_profile.portfolio.assets:
            if asset.type == "stock" and asset.metrics:
                # Usamos .evaluate() que é o novo nome no benchmark.py
                stock_analyses[asset.symbol] = StockStrategy.evaluate(asset.metrics)
                if user_profile.profile_plan != "free":
                    forecasts[asset.symbol] = PredictiveService.forecast_price(asset.symbol)

            elif asset.type == "fii" and asset.metrics:
                fii_analyses[asset.symbol] = FiiStrategy.evaluate(asset.metrics)
                if user_profile.profile_plan != "free":
                    forecasts[asset.symbol] = PredictiveService.forecast_price(asset.symbol)

        # 1. Se Free: retorna só análise estratégica
        if user_profile.profile_plan == "free":
            return {
                "plan": "free",
                "stock_scores": stock_analyses,
                "fii_scores": fii_analyses,
                "message": "Upgrade para Premium para análise com IA, Radar de Oportunidades e Detecção de Erros.",
                "timestamp": datetime.now().isoformat(),
            }

        # 2. Se Premium/Pro: IA faz análise completa (Score, Radar, Erros)
        prompt = AIAnalysisService.prepare_analysis_prompt(
            user_profile, stock_analyses, fii_analyses, forecasts
        )

        ai_response = await AIAnalysisService.analyze_with_ai(prompt)

        return {
            "plan": user_profile.user_id, # Usando ID para contexto
            "profile_plan": user_profile.profile_plan,
            "stock_scores": stock_analyses,
            "fii_scores": fii_analyses,
            "forecasts": forecasts,
            "ai_analysis": ai_response, # Novo nome para o payload completo
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        fastapi_logger.error(f"Erro na análise: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate")
async def simulate_portfolio(request: SimulationRequest):
    """
    Simulação de futuro baseada em aportes mensais
    """
    try:
        return SimulationService.simulate(request)
    except Exception as e:
        fastapi_logger.error(f"Erro na simulação: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.5.0", "engine": "Groq/Llama-3.3"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
