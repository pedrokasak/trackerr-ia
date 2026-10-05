"""
Checagens determinísticas da avaliação (TRA-242), sem LLM. Complementam o
juiz e valem mais que ele onde dá para checar em código:

- número da resposta que não existe no contexto nem na pergunta;
- linguagem de recomendação (o mesmo guardrail das respostas do RAG);
- disclaimer presente (o RAG anexa sempre; ausência é bug).

Só para respostas do RAG. As respostas do regex e do roteador com tools
são templates com o número vindo do cálculo; nelas "venda" aparece por
direito ("se você vender a posição…") e o guardrail daria falso alarme.
"""

import re
from dataclasses import dataclass
from typing import List

from rag.query_service import DISCLAIMER
from rag.response_guard import validate_rag_response
from ri.fidelity import number_set, unsupported_numbers

# Nota de frescor que o próprio código prefixa ("…cerca de 40 dias…"): o
# número dela não vem do modelo e não vale como alucinação.
_FRESHNESS_NOTE = re.compile(
    r"^(?:⚠️\s*Aten[cç][aã]o:|Observa[cç][aã]o:)[^\n]*\n*", re.IGNORECASE
)


@dataclass(frozen=True)
class DeterministicChecks:
    unsupported_numbers: List[str]
    recommendation_language: bool
    disclaimer_present: bool

    @property
    def numeric_hallucination(self) -> bool:
        return bool(self.unsupported_numbers)


def answer_body(answer: str) -> str:
    """A parte que o modelo escreveu: sem disclaimer e sem a nota de frescor."""
    body = str(answer or "").replace(DISCLAIMER, "").strip()
    return _FRESHNESS_NOTE.sub("", body).strip()


def check_rag_answer(question: str, context: str, answer: str) -> DeterministicChecks:
    body = answer_body(answer)
    allowed = number_set(context) | number_set(question)
    guard = validate_rag_response(body)
    return DeterministicChecks(
        unsupported_numbers=unsupported_numbers(body, allowed),
        recommendation_language=guard.reason == "recommendation_language",
        disclaimer_present=DISCLAIMER in str(answer or ""),
    )
