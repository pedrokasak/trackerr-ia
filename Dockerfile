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

# Coolify typically injects PORT; default to 8000.
CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

