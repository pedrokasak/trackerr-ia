"""
Interface abstrata para providers de LLM.
Qualquer novo provider deve herdar de LLMProvider e implementar o método analyze.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ToolSpec:
    """
    Ferramenta oferecida ao modelo (TRA-241): nome, descrição e o JSON
    schema dos argumentos. O modelo só escolhe; quem executa é o server.
    """

    name: str
    description: str
    parameters: Dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """Uma chamada escolhida pelo modelo, com os argumentos que ele preencheu."""

    name: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class ToolCallsResult:
    calls: List[ToolCall]
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0


class ToolCallingNotSupported(Exception):
    """
    O provider não tem tool-calling nativo (TRA-241). Não é falha do
    provider: quem chama passa para o próximo da cadeia ou para a rota
    sem ferramentas.
    """


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

    async def call_tools(
        self, system: str, prompt: str, tools: List[ToolSpec]
    ) -> ToolCallsResult:
        """
        Tool-calling nativo, de passo único (TRA-241): o modelo escolhe quais
        ferramentas chamar e com que argumentos. Não executa nada nem
        responde à pergunta.

        Implementado nos providers com suporte nativo no SDK (Claude, Gemini,
        OpenRouter). Os outros herdam este padrão e lançam
        `ToolCallingNotSupported`.
        """
        raise ToolCallingNotSupported(self.provider_name)

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Nome do provider para logging e identificação."""
        ...
