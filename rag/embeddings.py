"""
Geracao de embedding pro RAG (TRA-37).

Nao roteia por LLM_PROVIDER/LLMFactory: Claude e Groq nao tem API de
embedding (LLMProvider.analyze so cobre geracao de texto). Embedding usa
Gemini sempre, independente de qual provider esta configurado pra chat —
por isso exige GEMINI_API_KEY mesmo quando LLM_PROVIDER=claude ou groq.

Dimensao fixa em 768 pra bater com a coluna `embedding vector(768)` de
rag/models.py — trocar de modelo exige migracao, documentado nos dois
lugares de proposito.
"""

import os
from abc import ABC, abstractmethod

from google import genai

from rag.models import EMBEDDING_DIM

EMBEDDING_MODEL = "text-embedding-004"


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        ...


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY não configurada — necessária para embeddings "
                "mesmo quando LLM_PROVIDER não é gemini."
            )
        self._client = genai.Client(api_key=api_key)

    async def embed(self, text: str) -> list[float]:
        response = await self._client.aio.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[text],
        )
        if not response.embeddings or not response.embeddings[0].values:
            raise RuntimeError("Gemini não retornou embedding para o texto enviado.")

        values = response.embeddings[0].values
        if len(values) != EMBEDDING_DIM:
            raise RuntimeError(
                f"Embedding com dimensão inesperada: {len(values)} "
                f"(schema espera {EMBEDDING_DIM}). Modelo ou config mudou?"
            )
        return values
