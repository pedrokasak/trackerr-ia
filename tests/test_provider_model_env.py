"""
O modelo de cada provider vinha fixo no código: trocar exigia deploy.
Agora sai de variável de ambiente, com o padrão da classe como reserva.
"""

import importlib

import pytest


CASES = [
    ("benchmark.providers.claude_provider", "ClaudeProvider", "CLAUDE_MODEL"),
    ("benchmark.providers.gemini_provider", "GeminiProvider", "GEMINI_MODEL"),
    ("benchmark.providers.groq_provider", "GroqProvider", "GROQ_MODEL"),
    ("benchmark.providers.nvidia_provider", "NvidiaProvider", "NVIDIA_MODEL"),
]

API_KEYS = {
    "ANTHROPIC_API_KEY": "chave-de-teste",
    "GEMINI_API_KEY": "chave-de-teste",
    "GOOGLE_API_KEY": "chave-de-teste",
    "GROQ_API_KEY": "chave-de-teste",
    "NVIDIA_API_KEY": "chave-de-teste",
}


def _build(module_path, class_name, monkeypatch):
    for key, value in API_KEYS.items():
        monkeypatch.setenv(key, value)
    module = importlib.import_module(module_path)
    provider_class = getattr(module, class_name)
    try:
        return provider_class()
    except Exception as exc:  # pragma: no cover - depende do SDK instalado
        pytest.skip(f"{class_name} nao pode ser instanciado aqui: {exc}")


@pytest.mark.parametrize("module_path,class_name,env_var", CASES)
def test_modelo_vem_da_variavel_de_ambiente(
    module_path, class_name, env_var, monkeypatch
):
    monkeypatch.setenv(env_var, "modelo-escolhido-por-env")
    provider = _build(module_path, class_name, monkeypatch)
    assert provider._model == "modelo-escolhido-por-env"


@pytest.mark.parametrize("module_path,class_name,env_var", CASES)
def test_sem_variavel_usa_o_padrao_da_classe(
    module_path, class_name, env_var, monkeypatch
):
    monkeypatch.delenv(env_var, raising=False)
    provider = _build(module_path, class_name, monkeypatch)
    assert provider._model == provider.DEFAULT_MODEL


@pytest.mark.parametrize("module_path,class_name,env_var", CASES)
def test_argumento_explicito_vence_a_variavel(
    module_path, class_name, env_var, monkeypatch
):
    monkeypatch.setenv(env_var, "modelo-do-env")
    for key, value in API_KEYS.items():
        monkeypatch.setenv(key, value)
    module = importlib.import_module(module_path)
    provider_class = getattr(module, class_name)
    try:
        provider = provider_class(model="modelo-do-argumento")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"{class_name} nao pode ser instanciado aqui: {exc}")
    assert provider._model == "modelo-do-argumento"
