"""
Impressão digital dos prompts que geram texto para o usuário (TRA-242).

O dataset dourado guarda a impressão dos prompts com que foi gravado. Se um
prompt mudar, a impressão muda e o CI falha até alguém rodar a avaliação de
novo (`python -m evals.golden record`) e as notas passarem do limiar. É o
que transforma "trocar prompt sem medo" em regra: mudança de prompt sem
avaliação não entra.

Pega o código-fonte das funções que montam os prompts, então até mudança
de comentário nelas pede nova gravação — de propósito: é barato, e evita
discutir o que conta como mudança.
"""

import hashlib
import inspect

from rag.query_service import DISCLAIMER, RagQueryService
from ri.knowledge_service import RiDocumentAnswerer
from ri.summary_service import RiSummaryService

_PROMPT_SOURCES = (
    RagQueryService._build_prompt,
    RiSummaryService.prepare_prompt,
    RiDocumentAnswerer.prepare_prompt,
)


def prompt_fingerprint() -> str:
    digest = hashlib.sha256()
    for function in _PROMPT_SOURCES:
        digest.update(inspect.getsource(function).encode("utf-8"))
    digest.update(DISCLAIMER.encode("utf-8"))
    return digest.hexdigest()[:16]
