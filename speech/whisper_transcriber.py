"""
Transcrição por Whisper em APIs no formato da OpenAI (Groq e OpenRouter).

Os dois providers aceitam o mesmo multipart em `/audio/transcriptions`; o
que muda é a URL, a chave, o modelo e como cada um se comporta. O spike de
12/11/2026 (`docs/spikes/voz/`) mediu:

- Groq direto: ~0,5s, texto completo e `prompt` de tickers efetivo. É o
  padrão.
- OpenRouter: mais lento, ignora o `prompt` e às vezes devolve a
  transcrição cortada. Por isso confere `usage.seconds` contra a duração
  gravada e trata transcrição curta demais como provider indisponível.
"""

import os
from typing import Optional, Sequence

import httpx
from fastapi.logger import logger

from .base import SpeechProviderUnavailable, SpeechToTextProvider, Transcription

# O Cloudflare da Groq bloqueia cliente sem User-Agent (erro 1010).
USER_AGENT = "TrackerrIA/1.0"
_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
# Abaixo disso a transcrição cobriu só parte do áudio enviado.
_MIN_COVERAGE = 0.8
_BASE_VOCABULARY = "dividendos, proventos, carteira, CDI, Ibovespa"
# O prompt do Whisper tem teto de ~224 tokens; ticker vale ~3 a 4 tokens.
# Passando disso a Groq corta ou recusa, e a recusa empurra para o fallback
# que ignora o prompt. O server manda primeiro os tickers de maior peso.
_MAX_PROMPT_TICKERS = 40
# A Groq deduz o formato pela extensão do nome do arquivo e recusa nome sem
# extensão com 400 (visto no teste de ponta a ponta).
_EXTENSIONS = {
    "audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "m4a", "audio/m4a": "m4a",
    "audio/x-m4a": "m4a", "audio/aac": "m4a", "audio/mpeg": "mp3", "audio/wav": "wav", "audio/x-wav": "wav",
}


class WhisperTranscriber(SpeechToTextProvider):
    def __init__(
        self,
        provider_name: str,
        base_url: str,
        api_key_env: str,
        model_env: str,
        default_model: str,
        model: Optional[str] = None,
        check_coverage: bool = False,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} não configurada no .env")
        self._name = provider_name
        self._base_url = base_url
        self._api_key = api_key
        self._model = model or os.getenv(model_env) or default_model
        self._check_coverage = check_coverage
        self._transport = transport
        logger.info("[%s-stt] Modelo resolvido: %s", provider_name, self._model)

    @property
    def provider_name(self) -> str:
        return self._name

    async def transcribe(
        self,
        audio: bytes,
        mime_type: str,
        language: str,
        vocabulary: Sequence[str],
        expected_seconds: Optional[float] = None,
    ) -> Transcription:
        prompt = ", ".join([*list(vocabulary)[:_MAX_PROMPT_TICKERS], _BASE_VOCABULARY])
        data = {"model": self._model, "language": language, "response_format": "json", "prompt": prompt}
        files = {"file": (f"audio.{_EXTENSIONS.get(mime_type, 'mp3')}", audio, mime_type)}
        headers = {"Authorization": f"Bearer {self._api_key}", "User-Agent": USER_AGENT}
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, transport=self._transport) as client:
                response = await client.post(
                    f"{self._base_url}/audio/transcriptions", data=data, files=files, headers=headers
                )
        except httpx.HTTPError as exc:
            raise SpeechProviderUnavailable(f"{self._name}: {type(exc).__name__}") from exc
        if response.status_code != 200:
            # Só o status vai para o log: o corpo pode ecoar trecho da fala.
            raise SpeechProviderUnavailable(f"{self._name}: HTTP {response.status_code}")

        try:
            body = response.json()
        except ValueError as exc:
            raise SpeechProviderUnavailable(f"{self._name}: resposta sem JSON") from exc
        seconds = (body.get("usage") or {}).get("seconds")
        if self._check_coverage and expected_seconds and seconds is not None:
            if seconds < expected_seconds * _MIN_COVERAGE:
                raise SpeechProviderUnavailable(
                    f"{self._name}: transcrição cobriu {seconds:.1f}s de {expected_seconds:.1f}s"
                )
        return Transcription(
            text=(body.get("text") or "").strip(),
            provider=self._name,
            model=self._model,
            audio_seconds=seconds,
        )


def groq_transcriber(model: Optional[str] = None, **kwargs) -> WhisperTranscriber:
    return WhisperTranscriber(
        provider_name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        model_env="GROQ_STT_MODEL",
        default_model="whisper-large-v3-turbo",
        model=model,
        **kwargs,
    )


def openrouter_transcriber(model: Optional[str] = None, **kwargs) -> WhisperTranscriber:
    return WhisperTranscriber(
        provider_name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model_env="OPENROUTER_STT_MODEL",
        default_model="openai/whisper-large-v3-turbo",
        model=model,
        check_coverage=True,
        **kwargs,
    )
