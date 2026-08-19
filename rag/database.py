"""
Conexao com o Postgres proprio do RAG (TRA-35).

Banco dedicado ao trackerr-ia, independente do MongoDB transacional do
server — nunca acessa o banco transacional diretamente (principio 2 de
docs/rag-trackerr-ia.md). Sobrevive a qualquer decisao futura sobre onde o
Mongo roda (hoje Atlas, futuramente Coolify): este Postgres e infra propria
do trackerr-ia, hospedada onde for mais conveniente (inclusive o mesmo
Coolify, que tambem serve Postgres).
"""

import os
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


class RagDatabaseNotConfiguredError(RuntimeError):
    """Levantado quando RAG_DATABASE_URL nao esta setada."""


def _normalize_async_url(raw_url: str) -> str:
    """
    Aceita tanto `postgresql://...` quanto `postgresql+asyncpg://...` na env
    var — reescreve pro driver async se vier no formato "comum" copiado de
    uma connection string padrao (Coolify, Railway, Neon costumam gerar
    `postgresql://`).
    """
    if raw_url.startswith("postgresql+asyncpg://"):
        return raw_url
    if raw_url.startswith("postgresql://"):
        return raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if raw_url.startswith("postgres://"):
        return raw_url.replace("postgres://", "postgresql+asyncpg://", 1)
    return raw_url


def get_rag_database_url() -> str:
    raw_url = os.getenv("RAG_DATABASE_URL")
    if not raw_url:
        raise RagDatabaseNotConfiguredError(
            "RAG_DATABASE_URL nao configurada. Ver .env.example."
        )
    return _normalize_async_url(raw_url)


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(get_rag_database_url(), pool_pre_ping=True)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(), expire_on_commit=False
        )
    return _session_factory


async def get_rag_session() -> AsyncIterator[AsyncSession]:
    """Dependency FastAPI-style: `session = Depends(get_rag_session)`."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        yield session
