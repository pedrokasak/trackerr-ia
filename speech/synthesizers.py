"""
Síntese de voz: OpenRouter (padrão, Voxtral) e Gemini direto (fallback).

Escolhas do spike de 11 e 12/11/2026 (`docs/spikes/voz/`):

- Voxtral Mini TTS 3.0 pelo OpenRouter, voz `voice_3150`: escolhido no teste
  cego, com streaming real e primeiro byte em ~1s.
- Gemini direto: sem streaming no `generateContent` e só PCM. Entra como
  fallback; o PCM vira WAV para o navegador tocar pelo `Content-Type`.
"""

import base64
import os
import re
import struct
from typing import AsyncIterator, Optional

import httpx
from fastapi.logger import logger

from .base import SpeechProviderUnavailable, SynthesizedSpeech, TextToSpeechProvider
from .whisper_transcriber import USER_AGENT

_TIMEOUT = httpx.Timeout(60.0, connect=5.0)
_PCM_RATE = 24000


def pcm_to_wav(pcm: bytes, rate: int = _PCM_RATE) -> bytes:
    """PCM 16 bits mono com cabeçalho WAV, para o navegador tocar."""
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    return header + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm


async def _single_chunk(data: bytes) -> AsyncIterator[bytes]:
    yield data


class OpenRouterSynthesizer(TextToSpeechProvider):
    DEFAULT_MODEL = "mistralai/voxtral-mini-tts-3-0"
    DEFAULT_VOICE = "voice_3150"

    def __init__(
        self,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY não configurada no .env")
        self._api_key = api_key
        self._model = model or os.getenv("OPENROUTER_TTS_MODEL") or self.DEFAULT_MODEL
        # A voz pertence ao modelo: trocar só o modelo deixaria uma voz que
        # ele não conhece. Sem OPENROUTER_TTS_VOICE, o padrão só vale para o
        # modelo padrão.
        env_voice = os.getenv("OPENROUTER_TTS_VOICE")
        if voice or env_voice:
            self._voice = voice or env_voice
        elif self._model == self.DEFAULT_MODEL:
            self._voice = self.DEFAULT_VOICE
        else:
            raise ValueError("OPENROUTER_TTS_VOICE é obrigatória quando OPENROUTER_TTS_MODEL muda")
        # Gemini pelo OpenRouter só aceita PCM.
        self._format = "pcm" if self._model.startswith("google/") else "mp3"
        self._transport = transport
        logger.info("[openrouter-tts] Modelo %s, voz %s", self._model, self._voice)

    @property
    def provider_name(self) -> str:
        return "openrouter"

    async def synthesize(self, text: str) -> SynthesizedSpeech:
        client = httpx.AsyncClient(timeout=_TIMEOUT, transport=self._transport)
        request = client.build_request(
            "POST",
            "https://openrouter.ai/api/v1/audio/speech",
            json={"model": self._model, "input": text, "voice": self._voice, "response_format": self._format},
            headers={"Authorization": f"Bearer {self._api_key}", "User-Agent": USER_AGENT, "X-Title": "Trackerr IA"},
        )
        try:
            response = await client.send(request, stream=True)
        except httpx.HTTPError as exc:
            await client.aclose()
            raise SpeechProviderUnavailable(f"openrouter: {type(exc).__name__}") from exc
        if response.status_code != 200:
            await response.aclose()
            await client.aclose()
            raise SpeechProviderUnavailable(f"openrouter: HTTP {response.status_code}")

        if self._format == "pcm":
            try:
                pcm = await response.aread()
            finally:
                await response.aclose()
                await client.aclose()
            return SynthesizedSpeech("audio/wav", self.provider_name, self._model, self._voice, _single_chunk(pcm_to_wav(pcm)))

        async def stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()

        return SynthesizedSpeech("audio/mpeg", self.provider_name, self._model, self._voice, stream())


class GeminiSynthesizer(TextToSpeechProvider):
    DEFAULT_MODEL = "gemini-3.8-flash-tts"
    DEFAULT_VOICE = "Kore"

    def __init__(
        self,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY não configurada no .env")
        self._api_key = api_key
        self._model = model or os.getenv("GEMINI_TTS_MODEL") or self.DEFAULT_MODEL
        self._voice = voice or os.getenv("GEMINI_TTS_VOICE") or self.DEFAULT_VOICE
        self._transport = transport

    @property
    def provider_name(self) -> str:
        return "gemini"

    async def synthesize(self, text: str) -> SynthesizedSpeech:
        body = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self._voice}}},
            },
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, transport=self._transport) as client:
                response = await client.post(
                    url, json=body, headers={"x-goog-api-key": self._api_key, "User-Agent": USER_AGENT}
                )
        except httpx.HTTPError as exc:
            raise SpeechProviderUnavailable(f"gemini: {type(exc).__name__}") from exc
        if response.status_code != 200:
            raise SpeechProviderUnavailable(f"gemini: HTTP {response.status_code}")
        try:
            inline = response.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
            data = inline["data"]
        except (KeyError, IndexError, ValueError) as exc:
            raise SpeechProviderUnavailable("gemini: resposta sem áudio") from exc
        # O mimeType diz a taxa ("audio/L16;codec=pcm;rate=24000").
        rate = re.search(r"rate=(\d+)", inline.get("mimeType") or "")
        wav = pcm_to_wav(base64.b64decode(data), int(rate.group(1)) if rate else _PCM_RATE)
        return SynthesizedSpeech("audio/wav", self.provider_name, self._model, self._voice, _single_chunk(wav))
