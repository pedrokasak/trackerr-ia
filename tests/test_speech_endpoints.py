"""Endpoints /api/speech/* (ADR-007)."""

from typing import AsyncIterator

import pytest
from fastapi.testclient import TestClient

import main
from main import app
from speech.base import SpeechProviderUnavailable, SynthesizedSpeech, Transcription

client = TestClient(app)


class _FakeStt:
    provider_name = "groq"

    def __init__(self, text="A Petri4 pagou?", error=False):
        self.text = text
        self.error = error
        self.calls = []

    async def transcribe(self, audio, mime_type, language, vocabulary, expected_seconds=None):
        self.calls.append({"mime": mime_type, "vocabulary": vocabulary, "expected": expected_seconds, "size": len(audio)})
        if self.error:
            raise SpeechProviderUnavailable("groq fora")
        return Transcription(text=self.text, provider="groq", model="whisper-large-v3-turbo")


class _FakeTts:
    provider_name = "openrouter"

    def __init__(self, error=False):
        self.error = error

    async def synthesize(self, text):
        if self.error:
            raise SpeechProviderUnavailable("openrouter fora")

        async def chunks() -> AsyncIterator[bytes]:
            yield b"parte-1"
            yield b"parte-2"

        return SynthesizedSpeech("audio/mpeg", "openrouter", "mistralai/voxtral-mini-tts-3-0", "voice_3150", chunks())


@pytest.fixture
def stt(monkeypatch):
    fake = _FakeStt()
    monkeypatch.setattr(main.SpeechFactory, "get_stt", staticmethod(lambda: fake))
    return fake


def _post_audio(data=None, content_type="audio/mp4", payload=b"m4a-bytes"):
    return client.post(
        "/api/speech/transcribe",
        files={"file": ("pergunta.m4a", payload, content_type)},
        data=data or {"tickers": "petr4, ITUB4", "duration_ms": "5200"},
    )


def test_transcreve_e_corrige_ticker_da_carteira(stt):
    response = _post_audio()
    assert response.status_code == 200
    assert response.json() == {"text": "A PETR4 pagou?", "no_speech": False, "provider": "groq", "model": "whisper-large-v3-turbo"}
    assert stt.calls == [{"mime": "audio/mp4", "vocabulary": ["PETR4", "ITUB4"], "expected": 5.2, "size": 9}]


def test_aceita_webm_com_codecs(stt):
    assert _post_audio(content_type="audio/webm;codecs=opus").status_code == 200
    assert stt.calls[0]["mime"] == "audio/webm"


def test_recusa_tipo_que_nao_e_audio(stt):
    response = _post_audio(content_type="application/pdf")
    assert response.status_code == 415
    assert stt.calls == []


def test_recusa_audio_grande_demais(stt):
    response = _post_audio(payload=b"x" * (main.SPEECH_MAX_BYTES + 1))
    assert response.status_code == 413
    assert stt.calls == []


def test_silencio_vira_no_speech(monkeypatch):
    fake = _FakeStt(text=" Obrigado. ")
    monkeypatch.setattr(main.SpeechFactory, "get_stt", staticmethod(lambda: fake))
    assert _post_audio().json()["no_speech"] is True


def test_stt_indisponivel_vira_503(monkeypatch):
    monkeypatch.setattr(main.SpeechFactory, "get_stt", staticmethod(lambda: _FakeStt(error=True)))
    response = _post_audio()
    assert response.status_code == 503
    assert response.json() == {"detail": "speech_unavailable"}


def test_erro_de_validacao_em_multipart_nao_devolve_o_audio(stt):
    # Sem o campo `file`: o handler de validação não pode ecoar o corpo,
    # porque num upload de voz o corpo é a fala do usuário.
    response = client.post("/api/speech/transcribe", files={"outro": ("x.m4a", b"voz-secreta", "audio/mp4")})
    assert response.status_code == 422
    assert "voz-secreta" not in response.text
    assert "body" not in response.json()


def test_sintetiza_em_stream_com_cabecalhos(monkeypatch):
    monkeypatch.setattr(main.SpeechFactory, "get_tts", staticmethod(lambda: _FakeTts()))
    response = client.post("/api/speech/synthesize", json={"text": "A PETR4 pagou mil reais."})
    assert response.status_code == 200
    assert response.content == b"parte-1parte-2"
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.headers["x-speech-voice"] == "voice_3150"
    assert response.headers["cache-control"] == "no-store"


def test_sintese_recusa_texto_longo_demais(monkeypatch):
    monkeypatch.setattr(main.SpeechFactory, "get_tts", staticmethod(lambda: _FakeTts()))
    assert client.post("/api/speech/synthesize", json={"text": "a" * 1201}).status_code == 422


def test_tts_indisponivel_vira_503(monkeypatch):
    monkeypatch.setattr(main.SpeechFactory, "get_tts", staticmethod(lambda: _FakeTts(error=True)))
    response = client.post("/api/speech/synthesize", json={"text": "Olá"})
    assert response.status_code == 503


def test_audio_de_voz_nao_vai_para_disco():
    # Acima do limite em memória o Starlette grava o upload em arquivo temporário.
    from starlette.formparsers import MultiPartParser

    assert MultiPartParser.spool_max_size > main.SPEECH_MAX_BYTES
