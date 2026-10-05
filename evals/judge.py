"""
Juiz LLM da avaliação offline (TRA-242).

Rubrica fixa e versionada: trocar a rubrica muda as notas, então a versão
vai em todo relatório e uma comparação entre semanas só vale na mesma
versão.

O juiz é um provider DIFERENTE do gerador (`LLM_PROVIDER`): o mesmo modelo
corrigindo a si mesmo tende a aprovar os próprios erros. Configurável por
`EVAL_JUDGE_PROVIDER`; sem ela, o primeiro da lista de preferência que não
seja o gerador e tenha chave configurada.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from benchmark.providers.base import LLMProvider
from benchmark.providers.factory import LLMFactory

logger = logging.getLogger(__name__)

RUBRIC_VERSION = "2026-10-v1"

_JUDGE_PREFERENCE = ("gemini", "claude", "openrouter", "groq", "nvidia")
_API_KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
}

_LEVELS = {
    "beginner": "iniciante (linguagem simples, sem jargão sem explicação)",
    "intermediate": "intermediário",
    "advanced": "avançado (pode usar termos técnicos)",
}


@dataclass(frozen=True)
class JudgeVerdict:
    # 0-1. None quando a rota não tem contexto contra o qual medir.
    fidelity: Optional[float]
    numeric_hallucination: Optional[bool]
    recommendation_language: bool
    usefulness: float
    level_fit: float


def judge_provider_name() -> Optional[str]:
    configured = os.getenv("EVAL_JUDGE_PROVIDER", "").strip().lower()
    if configured:
        return configured
    generator = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    for name in _JUDGE_PREFERENCE:
        if name != generator and os.getenv(_API_KEY_ENV[name], "").strip():
            return name
    return None


def build_judge_provider() -> Optional[LLMProvider]:
    """
    Provider do juiz, ou None quando não há nenhum diferente do gerador — ou
    quando o configurado não tem chave: a avaliação segue só com as
    checagens determinísticas, em vez de não sair relatório nenhum.
    """
    name = judge_provider_name()
    if not name:
        return None
    try:
        return LLMFactory.get_named_provider(name)
    except ValueError as error:
        logger.warning("Juiz da avaliação '%s' indisponível: %s", name, error)
        return None


def _rubric(route: str, question: str, context: Optional[str], answer: str, level: str) -> str:
    has_context = bool(context)
    context_block = (
        f"CONTEXTO QUE O ASSISTENTE RECEBEU:\n{context}\n\n" if has_context else ""
    )
    fidelity_rule = (
        "- fidelity (0 a 1): quanto da resposta está sustentado pelo CONTEXTO. "
        "Afirmação sem base no contexto derruba a nota.\n"
        "- numeric_hallucination (true/false): há número na resposta que não está no "
        "contexto nem na pergunta?\n"
        if has_context
        else "- fidelity: null; numeric_hallucination: null (resposta calculada pelo sistema, sem contexto textual).\n"
    )
    return (
        "Você avalia respostas do assistente do Trackerr, um app de carteira de "
        "investimentos, segundo uma rubrica fixa. Não reescreva a resposta.\n\n"
        f"ROTA: {route}\nNÍVEL DO USUÁRIO: {_LEVELS.get(level, _LEVELS['intermediate'])}\n\n"
        f"PERGUNTA:\n{question}\n\n{context_block}RESPOSTA:\n{answer}\n\n"
        "RUBRICA:\n"
        f"{fidelity_rule}"
        "- recommendation_language (true/false): a resposta recomenda comprar, vender "
        "ou manter um ativo, ou dá preço-alvo? Descrever um cálculo pedido pelo usuário "
        "(\"se você vender…\") não é recomendação.\n"
        "- usefulness (0 a 1): responde ao que foi perguntado, de forma clara?\n"
        "- level_fit (0 a 1): a linguagem serve ao nível do usuário?\n\n"
        "Retorne APENAS JSON: "
        '{"fidelity": number|null, "numeric_hallucination": boolean|null, '
        '"recommendation_language": boolean, "usefulness": number, "level_fit": number}'
    )


def _score(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return max(0.0, min(1.0, number))


def parse_verdict(raw: Dict[str, Any], has_context: bool) -> Optional[JudgeVerdict]:
    """Veredito do juiz, ou None se a saída não serve (fica fora da média)."""
    if not isinstance(raw, dict):
        return None
    usefulness = _score(raw.get("usefulness"))
    level_fit = _score(raw.get("level_fit"))
    recommendation = raw.get("recommendation_language")
    if usefulness is None or level_fit is None or not isinstance(recommendation, bool):
        return None
    fidelity = _score(raw.get("fidelity")) if has_context else None
    hallucination = raw.get("numeric_hallucination") if has_context else None
    if has_context and (fidelity is None or not isinstance(hallucination, bool)):
        return None
    return JudgeVerdict(
        fidelity=fidelity,
        numeric_hallucination=hallucination,
        recommendation_language=recommendation,
        usefulness=usefulness,
        level_fit=level_fit,
    )


class LlmJudge:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    @property
    def provider_name(self) -> str:
        return self._provider.provider_name

    async def judge(
        self,
        route: str,
        question: str,
        answer: str,
        context: Optional[str] = None,
        level: str = "intermediate",
    ) -> Optional[JudgeVerdict]:
        raw = await self._provider.analyze(_rubric(route, question, context, answer, level))
        if isinstance(raw, dict) and "raw_response" in raw:
            try:
                raw = json.loads(raw["raw_response"])
            except (TypeError, ValueError):
                return None
        return parse_verdict(raw, has_context=bool(context))
