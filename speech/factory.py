"""
Factory da voz (ADR-007, decisão 2), no mesmo padrão do `LLMFactory`.

Variáveis:
  STT_PROVIDER          groq (padrão) | openrouter
  STT_PROVIDER_FALLBACK lista em ordem, separada por vírgula (ex.: openrouter)
  TTS_PROVIDER          openrouter (padrão) | gemini
  TTS_PROVIDER_FALLBACK lista em ordem (ex.: gemini)

Modelo e voz ficam em variáveis por provider (GROQ_STT_MODEL,
OPENROUTER_STT_MODEL, OPENROUTER_TTS_MODEL, OPENROUTER_TTS_VOICE,
GEMINI_TTS_MODEL, GEMINI_TTS_VOICE), separadas das do chat: trocar o modelo
do chat não muda a voz.

O fallback só cobre "o provider não respondeu". Provider com nome
desconhecido falha no boot com ValueError; fallback igual ao primário é
ignorado com aviso, como no `LLMFactory`.
"""

import os
from typing import Callable, Dict, List, Optional, Sequence

from fastapi.logger import logger

from .base import (
    SpeechProviderUnavailable,
    SpeechToTextProvider,
    SynthesizedSpeech,
    TextToSpeechProvider,
    Transcription,
)
from .synthesizers import GeminiSynthesizer, OpenRouterSynthesizer
from .whisper_transcriber import groq_transcriber, openrouter_transcriber

_STT_BUILDERS: Dict[str, Callable[[], SpeechToTextProvider]] = {
    "groq": groq_transcriber,
    "openrouter": openrouter_transcriber,
}
_TTS_BUILDERS: Dict[str, Callable[[], TextToSpeechProvider]] = {
    "openrouter": OpenRouterSynthesizer,
    "gemini": GeminiSynthesizer,
}


def _chain(primary_env: str, default: str, fallback_env: str, supported: Sequence[str]) -> List[str]:
    primary = (os.getenv(primary_env) or default).strip().lower()
    names = [primary]
    for item in (os.getenv(fallback_env) or "").split(","):
        name = item.strip().lower()
        if not name or name in names:
            if name == primary:
                logger.warning("[SpeechFactory] %s contém o próprio %s ('%s'); ignorado.", fallback_env, primary_env, name)
            continue
        names.append(name)
    unknown = [n for n in names if n not in supported]
    if unknown:
        raise ValueError(
            f"Provider de voz não suportado: {', '.join(unknown)}. Valores aceitos: {', '.join(supported)}"
        )
    return names


class FallbackTranscriber(SpeechToTextProvider):
    def __init__(self, providers: List[SpeechToTextProvider]) -> None:
        self._providers = providers

    @property
    def provider_name(self) -> str:
        return self._providers[0].provider_name

    async def transcribe(self, audio, mime_type, language, vocabulary, expected_seconds=None) -> Transcription:
        last: Optional[Exception] = None
        for index, provider in enumerate(self._providers):
            try:
                result = await provider.transcribe(audio, mime_type, language, vocabulary, expected_seconds)
            except SpeechProviderUnavailable as exc:
                logger.warning("[SpeechFactory] STT indisponível: %s", exc)
                last = exc
                continue
            if index > 0:
                logger.warning("[SpeechFactory] STT pelo fallback '%s'", provider.provider_name)
            return result
        raise SpeechProviderUnavailable(f"nenhum provider de STT respondeu ({last})")


class FallbackSynthesizer(TextToSpeechProvider):
    def __init__(self, providers: List[TextToSpeechProvider]) -> None:
        self._providers = providers

    @property
    def provider_name(self) -> str:
        return self._providers[0].provider_name

    async def synthesize(self, text: str) -> SynthesizedSpeech:
        last: Optional[Exception] = None
        for index, provider in enumerate(self._providers):
            try:
                result = await provider.synthesize(text)
            except SpeechProviderUnavailable as exc:
                logger.warning("[SpeechFactory] TTS indisponível: %s", exc)
                last = exc
                continue
            if index > 0:
                # A voz muda no fallback; isso é aceito (ADR-007) e precisa
                # aparecer no log.
                logger.warning("[SpeechFactory] TTS pelo fallback '%s', voz '%s'", provider.provider_name, result.voice)
            return result
        raise SpeechProviderUnavailable(f"nenhum provider de TTS respondeu ({last})")


def _build_all(names: List[str], builders: Dict[str, Callable]) -> list:
    # Chave ausente conta como provider indisponível, inclusive no primário,
    # como no FallbackLLMProvider: a cadeia segue para o próximo.
    built = []
    for name in names:
        try:
            built.append(builders[name]())
        except ValueError as exc:
            logger.warning("[SpeechFactory] provider '%s' ignorado: %s", name, exc)
    if not built:
        raise ValueError(f"Nenhum provider de voz configurado entre: {', '.join(names)}")
    return built


class SpeechFactory:
    @staticmethod
    def get_stt() -> SpeechToTextProvider:
        names = _chain("STT_PROVIDER", "groq", "STT_PROVIDER_FALLBACK", tuple(_STT_BUILDERS))
        providers = _build_all(names, _STT_BUILDERS)
        logger.info("[SpeechFactory] STT: %s", ", ".join(p.provider_name for p in providers))
        return providers[0] if len(providers) == 1 else FallbackTranscriber(providers)

    @staticmethod
    def get_tts() -> TextToSpeechProvider:
        names = _chain("TTS_PROVIDER", "openrouter", "TTS_PROVIDER_FALLBACK", tuple(_TTS_BUILDERS))
        providers = _build_all(names, _TTS_BUILDERS)
        logger.info("[SpeechFactory] TTS: %s", ", ".join(p.provider_name for p in providers))
        return providers[0] if len(providers) == 1 else FallbackSynthesizer(providers)
