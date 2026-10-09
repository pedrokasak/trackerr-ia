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
   | `LLM_PROVIDER`        | Provider de LLM ativo: `gemini` (padrão), `claude`, `groq`, `nvidia` ou `openrouter` |
   | `GEMINI_API_KEY`      | Necessária quando `LLM_PROVIDER=gemini`                    |
   | `ANTHROPIC_API_KEY`   | Necessária quando `LLM_PROVIDER=claude`                    |
   | `GROQ_API_KEY`        | Necessária quando `LLM_PROVIDER=groq`                       |
   | `NVIDIA_API_KEY`      | Necessária quando `LLM_PROVIDER=nvidia`                     |
   | `OPENROUTER_API_KEY`  | Necessária quando `LLM_PROVIDER=openrouter`                 |
   | `LLM_PROVIDER_FALLBACK` | Opcional (TRA-203). Provider para o qual cai automaticamente se o primário falhar — a chave dele também precisa estar configurada |
   | `RAG_DATABASE_URL`    | Postgres do vector store (TRA-35) — ver seção RAG abaixo    |

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

## RAG — vector store (TRA-35)

Banco Postgres próprio do trackerr-ia (extensão `pgvector`), independente
do MongoDB transacional do `server`. Arquitetura completa em
[`docs/rag-trackerr-ia.md`](../docs/rag-trackerr-ia.md) no repo raiz.

1. Suba um Postgres 15+ com a extensão `vector` disponível (a migração cria
   a extensão automaticamente se o usuário do banco tiver permissão).
2. Configure `RAG_DATABASE_URL` no `.env` (aceita `postgresql://` ou
   `postgresql+asyncpg://`).
3. Rode as migrações:

   ```bash
   poetry run alembic upgrade head
   ```

Toda query de retrieval (`rag/repository.py`) exige `user_id` e filtra por
ele sempre — nunca opcional. Ver `tests/test_rag_repository.py` para o
teste que trava esse invariante.
