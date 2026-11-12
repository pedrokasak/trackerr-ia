"""Adapters de STT e TTS e o SpeechFactory (ADR-007, decisão 2)."""

import base64
import json

import httpx
import pytest

from speech.base import SpeechProviderUnavailable, SynthesizedSpeech, Transcription
from speech.factory import FallbackSynthesizer, FallbackTranscriber, SpeechFactory
from speech.synthesizers import GeminiSynthesizer, OpenRouterSynthesizer, pcm_to_wav
from speech.whisper_transcriber import groq_transcriber, openrouter_transcriber


@pytest.fixture
def keys(monkeypatch):
    for name in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(name, "test-key")
    for name in (
        "STT_PROVIDER", "STT_PROVIDER_FALLBACK", "TTS_PROVIDER", "TTS_PROVIDER_FALLBACK",
        "GROQ_STT_MODEL", "OPENROUTER_STT_MODEL", "OPENROUTER_TTS_MODEL", "OPENROUTER_TTS_VOICE",
        "GEMINI_TTS_MODEL", "GEMINI_TTS_VOICE",
    ):
        monkeypatch.delenv(name, raising=False)


def _transport(handler):
    return httpx.MockTransport(handler)


# ---- STT ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_groq_manda_prompt_com_tickers_e_user_agent(keys):
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["ua"] = request.headers.get("user-agent")
        seen["body"] = request.content.decode("utf-8", "ignore")
        return httpx.Response(200, json={"text": " A PETR4 pagou? "})

    stt = groq_transcriber(transport=_transport(handler))
    result = await stt.transcribe(b"audio", "audio/mp4", "pt", ["PETR4", "ITUB4"])

    assert result == Transcription(text="A PETR4 pagou?", provider="groq", model="whisper-large-v3-turbo", audio_seconds=None)
    assert seen["url"].startswith("https://api.groq.com/openai/v1/audio/transcriptions")
    # Sem User-Agent o Cloudflare da Groq bloqueia (erro 1010).
    assert seen["ua"] == "TrackerrIA/1.0"
    assert "PETR4, ITUB4" in seen["body"]
    assert 'name="language"' in seen["body"]
    # Sem extensão no nome do arquivo a Groq responde 400.
    assert 'filename="audio.m4a"' in seen["body"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mime, ext", [("audio/webm", "webm"), ("audio/mpeg", "mp3"), ("audio/wav", "wav")])
async def test_nome_do_arquivo_segue_o_tipo_do_audio(keys, mime, ext):
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = request.content.decode("utf-8", "ignore")
        return httpx.Response(200, json={"text": "ok"})

    await groq_transcriber(transport=_transport(handler)).transcribe(b"audio", mime, "pt", [])
    assert f'filename="audio.{ext}"' in seen["body"]


@pytest.mark.asyncio
async def test_stt_erro_http_vira_indisponivel(keys):
    stt = groq_transcriber(transport=_transport(lambda r: httpx.Response(500, text="falha")))
    with pytest.raises(SpeechProviderUnavailable, match="HTTP 500"):
        await stt.transcribe(b"audio", "audio/mp4", "pt", [])


@pytest.mark.asyncio
async def test_openrouter_recusa_transcricao_truncada(keys):
    # Caso real do spike: a amostra de 21s voltou lida só até 10,5s.
    body = {"text": "A Petrae 4 pagou", "usage": {"seconds": 10.556}}
    stt = openrouter_transcriber(transport=_transport(lambda r: httpx.Response(200, json=body)))
    with pytest.raises(SpeechProviderUnavailable, match="cobriu"):
        await stt.transcribe(b"audio", "audio/mpeg", "pt", [], expected_seconds=21.2)


@pytest.mark.asyncio
async def test_openrouter_aceita_transcricao_completa(keys):
    body = {"text": "A PETR4 pagou", "usage": {"seconds": 20.9}}
    stt = openrouter_transcriber(transport=_transport(lambda r: httpx.Response(200, json=body)))
    result = await stt.transcribe(b"audio", "audio/mpeg", "pt", [], expected_seconds=21.2)
    assert result.provider == "openrouter"
    assert result.audio_seconds == 20.9


def test_stt_sem_chave_falha_no_boot(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        groq_transcriber()


def test_modelo_de_stt_vem_do_env(keys, monkeypatch):
    monkeypatch.setenv("GROQ_STT_MODEL", "whisper-large-v3")
    assert groq_transcriber()._model == "whisper-large-v3"


# ---- TTS ---------------------------------------------------------------------


async def _collect(speech: SynthesizedSpeech) -> bytes:
    return b"".join([c async for c in speech.chunks])


@pytest.mark.asyncio
async def test_openrouter_tts_usa_voxtral_e_faz_stream(keys):
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"mp3-bytes", headers={"content-type": "audio/mpeg"})

    speech = await OpenRouterSynthesizer(transport=_transport(handler)).synthesize("Olá")
    assert seen["body"] == {"model": "mistralai/voxtral-mini-tts-3-0", "input": "Olá", "voice": "voice_3150", "response_format": "mp3"}
    assert (speech.mime_type, speech.voice) == ("audio/mpeg", "voice_3150")
    assert await _collect(speech) == b"mp3-bytes"


@pytest.mark.asyncio
async def test_openrouter_tts_gemini_pede_pcm_e_devolve_wav(keys, monkeypatch):
    monkeypatch.setenv("OPENROUTER_TTS_MODEL", "google/gemini-3.8-flash-tts")
    monkeypatch.setenv("OPENROUTER_TTS_VOICE", "Kore")
    seen = {}

    def handler(request: httpx.Request):
        seen["format"] = json.loads(request.content)["response_format"]
        return httpx.Response(200, content=b"\x00\x01" * 10)

    speech = await OpenRouterSynthesizer(transport=_transport(handler)).synthesize("Olá")
    audio = await _collect(speech)
    assert seen["format"] == "pcm"
    assert speech.mime_type == "audio/wav"
    assert audio.startswith(b"RIFF") and audio.endswith(b"\x00\x01" * 10)


def test_trocar_modelo_de_tts_sem_voz_falha(keys, monkeypatch):
    # A voz pertence ao modelo: o padrão `voice_3150` não existe em outro.
    monkeypatch.setenv("OPENROUTER_TTS_MODEL", "elevenlabs/eleven-flash-v2.5")
    with pytest.raises(ValueError, match="OPENROUTER_TTS_VOICE"):
        OpenRouterSynthesizer()


@pytest.mark.asyncio
async def test_openrouter_tts_erro_http_vira_indisponivel(keys):
    synth = OpenRouterSynthesizer(transport=_transport(lambda r: httpx.Response(402, text="sem saldo")))
    with pytest.raises(SpeechProviderUnavailable, match="HTTP 402"):
        await synth.synthesize("Olá")


@pytest.mark.asyncio
async def test_gemini_direto_converte_pcm_em_wav(keys):
    pcm = b"\x10\x20" * 50
    payload = {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "audio/L16", "data": base64.b64encode(pcm).decode()}}]}}]}
    speech = await GeminiSynthesizer(transport=_transport(lambda r: httpx.Response(200, json=payload))).synthesize("Olá")
    assert (speech.provider, speech.voice, speech.mime_type) == ("gemini", "Kore", "audio/wav")
    assert await _collect(speech) == pcm_to_wav(pcm)


@pytest.mark.asyncio
async def test_gemini_sem_audio_vira_indisponivel(keys):
    synth = GeminiSynthesizer(transport=_transport(lambda r: httpx.Response(200, json={"candidates": []})))
    with pytest.raises(SpeechProviderUnavailable, match="sem áudio"):
        await synth.synthesize("Olá")


def test_pcm_to_wav_tem_cabecalho_correto():
    wav = pcm_to_wav(b"\x00" * 48000)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    assert len(wav) == 44 + 48000


# ---- Factory e fallback ------------------------------------------------------


def test_factory_padrao_e_groq_e_openrouter(keys):
    assert SpeechFactory.get_stt().provider_name == "groq"
    assert SpeechFactory.get_tts().provider_name == "openrouter"


def test_factory_monta_cadeia_de_fallback(keys, monkeypatch):
    monkeypatch.setenv("STT_PROVIDER_FALLBACK", "openrouter")
    monkeypatch.setenv("TTS_PROVIDER_FALLBACK", "gemini, openrouter")
    stt = SpeechFactory.get_stt()
    tts = SpeechFactory.get_tts()
    assert isinstance(stt, FallbackTranscriber)
    assert [p.provider_name for p in stt._providers] == ["groq", "openrouter"]
    # O primário repetido na lista de fallback é ignorado.
    assert [p.provider_name for p in tts._providers] == ["openrouter", "gemini"]


def test_factory_provider_desconhecido_falha(keys, monkeypatch):
    monkeypatch.setenv("STT_PROVIDER", "whisperx")
    with pytest.raises(ValueError, match="whisperx"):
        SpeechFactory.get_stt()


def test_fallback_sem_chave_nao_derruba_o_boot(keys, monkeypatch):
    monkeypatch.setenv("TTS_PROVIDER_FALLBACK", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY")
    assert SpeechFactory.get_tts().provider_name == "openrouter"


class _Fake:
    def __init__(self, name, error=False):
        self.provider_name = name
        self.error = error

    async def transcribe(self, *args, **kwargs):
        if self.error:
            raise SpeechProviderUnavailable(self.provider_name + " fora")
        return Transcription(text="ok", provider=self.provider_name, model="m")

    async def synthesize(self, text):
        if self.error:
            raise SpeechProviderUnavailable(self.provider_name + " fora")
        return SynthesizedSpeech("audio/mpeg", self.provider_name, "m", "v", None)


@pytest.mark.asyncio
async def test_fallback_de_stt_usa_o_proximo():
    result = await FallbackTranscriber([_Fake("groq", error=True), _Fake("openrouter")]).transcribe(b"", "audio/mp4", "pt", [])
    assert result.provider == "openrouter"


@pytest.mark.asyncio
async def test_fallback_de_tts_esgotado_levanta_indisponivel():
    with pytest.raises(SpeechProviderUnavailable, match="nenhum provider de TTS"):
        await FallbackSynthesizer([_Fake("openrouter", error=True), _Fake("gemini", error=True)]).synthesize("Olá")


@pytest.mark.asyncio
async def test_prompt_limita_a_quantidade_de_tickers(keys):
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = request.content.decode("utf-8", "ignore")
        return httpx.Response(200, json={"text": "ok"})

    tickers = [f"TK{chr(65 + i // 26)}{chr(65 + i % 26)}3" for i in range(100)]
    await groq_transcriber(transport=_transport(handler)).transcribe(b"a", "audio/mp4", "pt", tickers)
    assert tickers[39] in seen["body"]
    assert tickers[40] not in seen["body"]


@pytest.mark.asyncio
async def test_resposta_200_sem_json_vira_indisponivel(keys):
    stt = groq_transcriber(transport=_transport(lambda r: httpx.Response(200, text="<html>")))
    with pytest.raises(SpeechProviderUnavailable, match="sem JSON"):
        await stt.transcribe(b"audio", "audio/mp4", "pt", [])


def test_primario_sem_chave_cai_para_o_fallback(keys, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    monkeypatch.setenv("STT_PROVIDER_FALLBACK", "openrouter")
    assert SpeechFactory.get_stt().provider_name == "openrouter"


def test_nenhum_provider_com_chave_falha(keys, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(ValueError, match="Nenhum provider"):
        SpeechFactory.get_stt()


@pytest.mark.asyncio
async def test_gemini_respeita_a_taxa_do_mimetype(keys):
    pcm = b"\x00\x01" * 8
    payload = {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "audio/L16;codec=pcm;rate=16000", "data": base64.b64encode(pcm).decode()}}]}}]}
    speech = await GeminiSynthesizer(transport=_transport(lambda r: httpx.Response(200, json=payload))).synthesize("Olá")
    assert await _collect(speech) == pcm_to_wav(pcm, 16000)
