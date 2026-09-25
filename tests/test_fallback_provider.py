import pytest
from fastapi import HTTPException

from benchmark.providers.fallback_provider import FallbackLLMProvider


class FakeProvider:
    """Provider falso controlável — não é `LLMProvider` de verdade porque o
    protocolo é estrutural aqui (duck typing), igual o resto do módulo faz.
    """

    def __init__(self, name: str, result=None, error: Exception | None = None):
        self._name = name
        self._result = result
        self._error = error
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return self._name

    async def analyze(self, prompt: str):
        self.calls += 1
        if self._error:
            raise self._error
        return self._result


@pytest.mark.asyncio
async def test_primario_funciona_fallback_nunca_e_chamado():
    primary = FakeProvider("primario", result={"ok": True})
    fallback = FakeProvider("fallback", result={"ok": "nunca deveria aparecer"})
    provider = FallbackLLMProvider(primary, fallback)

    result = await provider.analyze("prompt")

    assert result == {"ok": True}
    assert primary.calls == 1
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_primario_falha_com_http_exception_fallback_assume():
    primary = FakeProvider(
        "primario", error=HTTPException(status_code=500, detail="fora do ar")
    )
    fallback = FakeProvider("fallback", result={"ok": True, "de": "fallback"})
    provider = FallbackLLMProvider(primary, fallback)

    result = await provider.analyze("prompt")

    assert result == {"ok": True, "de": "fallback"}
    assert primary.calls == 1
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_primario_falha_com_chave_ausente_fallback_assume():
    # ValueError e o que os construtores dos providers lancam quando a
    # API key nao esta configurada — chave ausente conta como "indisponivel".
    primary = FakeProvider(
        "primario", error=ValueError("API_KEY não configurada no .env")
    )
    fallback = FakeProvider("fallback", result={"ok": True})
    provider = FallbackLLMProvider(primary, fallback)

    result = await provider.analyze("prompt")

    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_ambos_falham_erro_sobe_normalmente():
    primary = FakeProvider(
        "primario", error=HTTPException(status_code=500, detail="primario caiu")
    )
    fallback = FakeProvider(
        "fallback", error=HTTPException(status_code=500, detail="fallback tambem caiu")
    )
    provider = FallbackLLMProvider(primary, fallback)

    with pytest.raises(HTTPException) as exc_info:
        await provider.analyze("prompt")

    # O erro que sobe e o do FALLBACK — e a ultima tentativa real que falhou.
    assert "fallback tambem caiu" in str(exc_info.value.detail)
    assert primary.calls == 1
    assert fallback.calls == 1


def test_provider_name_reflete_o_primario():
    primary = FakeProvider("primario")
    fallback = FakeProvider("fallback")
    provider = FallbackLLMProvider(primary, fallback)

    assert provider.provider_name == "primario"
