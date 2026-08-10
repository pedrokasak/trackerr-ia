# trakker-ia

FastAPI service used by Trackerr for AI analysis and simulations.

## Requisitos

- Python >= 3.11
- [Poetry](https://python-poetry.org/)

## Configuração

1. Instale as dependências:

   ```bash
   poetry install
   ```

2. Copie o `.env.example` para `.env` (se ainda não existir) e preencha as chaves:

   ```bash
   cp .env.example .env
   ```

   Variáveis disponíveis:

   | Variável             | Descrição                                              |
   | -------------------- | -------------------------------------------------------- |
   | `LLM_PROVIDER`        | Provider de LLM ativo: `gemini` (padrão), `claude` ou `groq` |
   | `GEMINI_API_KEY`      | Necessária quando `LLM_PROVIDER=gemini`                    |
   | `ANTHROPIC_API_KEY`   | Necessária quando `LLM_PROVIDER=claude`                    |
   | `GROQ_API_KEY`        | Necessária quando `LLM_PROVIDER=groq`                       |

   Preencha apenas a chave do provider selecionado.

## Rodando o serviço

```bash
poetry run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Ou, alternativamente:

```bash
poetry run python main.py
```

A API sobe em `http://localhost:8000`. Verifique com:

```bash
curl http://localhost:8000/api/health
```

## Testes

```bash
poetry run pytest
```
