"""
Interface abstrata para providers de LLM.
Qualquer novo provider deve herdar de LLMProvider e implementar o método analyze.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class LLMProvider(ABC):
    """Contrato base para todos os providers de LLM."""

    @abstractmethod
    async def analyze(self, prompt: str) -> Dict[str, Any]:
        """
        Envia o prompt para o LLM e retorna a resposta parseada como dict.

        Args:
            prompt: Texto com o contexto e instruções para análise.

        Returns:
            Dict com o resultado da análise (preferencialmente JSON parseado).
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Nome do provider para logging e identificação."""
        ...
