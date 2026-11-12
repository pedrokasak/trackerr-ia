"""
Contratos da voz (ADR-007, decisão 2).

`LLMProvider.analyze(prompt)` não serve para áudio, então a voz tem portas
próprias. Os adapters de cada provider implementam uma delas; quem escolhe o
adapter é o `SpeechFactory`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional, Sequence


class SpeechProviderUnavailable(Exception):
    """
    O provider não respondeu como deveria (erro de API, timeout, chave
    ausente, transcrição truncada). Quem chama passa para o próximo da
    cadeia de fallback.
    """


@dataclass(frozen=True)
class Transcription:
    text: str
    provider: str
    model: str
    # Segundos de áudio que o provider diz ter lido. Nem todo provider
    # devolve; o OpenRouter devolve em `usage.seconds`.
    audio_seconds: Optional[float] = None


@dataclass(frozen=True)
class SynthesizedSpeech:
    """
    Áudio pronto para ser repassado em stream. `chunks` só começa a ser
    consumido depois que o provider respondeu com sucesso, então uma falha
    de provider sempre acontece antes do primeiro byte chegar ao cliente.
    """

    mime_type: str
    provider: str
    model: str
    voice: str
    chunks: AsyncIterator[bytes]


class SpeechToTextProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        mime_type: str,
        language: str,
        vocabulary: Sequence[str],
        expected_seconds: Optional[float] = None,
    ) -> Transcription:
        """
        Transcreve o áudio. `vocabulary` vira o `prompt` do Whisper (tickers
        da carteira do usuário). `expected_seconds` é a duração gravada pelo
        cliente, usada para detectar transcrição truncada.
        """


class TextToSpeechProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    async def synthesize(self, text: str) -> SynthesizedSpeech:
        """Abre a síntese e devolve o áudio em stream."""
