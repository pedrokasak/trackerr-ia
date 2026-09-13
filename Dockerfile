FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# Prophet/pandas/numpy may need native deps; keep this minimal but practical.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --no-ansi --no-root

COPY . .

EXPOSE 8000

# Migra o banco do RAG antes de servir. Sem isto um Postgres recem-criado
# fica sem a extensao `vector` e sem as tabelas do alembic, e todo endpoint
# de RAG quebra em runtime com a API aparentemente saudavel. `set -e` via
# `&&`: migracao que falha impede a subida, em vez de servir com schema
# pela metade. Coolify injeta PORT; default 8000.
CMD ["bash", "-lc", "alembic upgrade head && exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

