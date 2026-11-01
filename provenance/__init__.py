"""
Proveniência das saídas de IA (ADR-004, seção 9.2).

Cada chamada ao LLM que o usuário vê ganha um bloco `provenance`: qual prompt,
qual modelo respondeu de fato, de que dados saiu, o que as checagens fizeram e
o hash do que foi entregue. Só ids e hashes — nunca o texto do prompt, da
carteira ou da resposta. Quem grava e sela o registro é o server, que sabe
quem é o usuário; aqui só se descreve o que aconteceu.

O bloco prova que *aquele resultado saiu daquele conjunto de dados, prompt e
modelo*. Não prova que o resultado está certo: isso é medido pelos evals.

O módulo só depende da biblioteca padrão e do Pydantic, para `models.models`
poder importá-lo sem ciclo.
"""

import hashlib
import json
import uuid
from datetime import date, datetime, timezone
import os
from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field

PROVENANCE_VERSION = 1

# Resultado do guarda numérico. `unchecked` é uma saída que nenhum guarda
# confere (a análise híbrida em JSON livre); dizer isso é mais honesto que "ok".
GuardOutcome = Literal[
    "ok", "number_hallucination", "llm_error", "not_called", "unchecked"
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


def canonical_json(value: Any) -> str:
    """JSON estável: chaves ordenadas, sem espaço, UTF-8 sem escape."""
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_default,
    )


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def new_generation_id() -> str:
    return "gen_" + uuid.uuid4().hex


class PromptRef(BaseModel):
    """O prompt enviado. `sha256` é do texto exato, então prova a entrada."""

    id: str
    version: str
    sha256: str


class ModelRef(BaseModel):
    """`requested` é o que a configuração pediu; `served_*` é quem respondeu."""

    requested: Optional[str] = None
    served_provider: Optional[str] = None
    model: Optional[str] = None


class DataWindow(BaseModel):
    """Menor e maior data das fontes usadas. Nulo quando nenhuma fonte tem data."""

    # `from` é palavra reservada em Python.
    from_: Optional[str] = Field(default=None, alias="from")
    to: Optional[str] = None

    model_config = {"populate_by_name": True}


class ProvenanceSource(BaseModel):
    source_type: Optional[str] = None
    source_id: Optional[str] = None
    knowledge_base: Optional[str] = None
    as_of: Optional[str] = None


class ProvenanceInputs(BaseModel):
    portfolio_snapshot_sha256: Optional[str] = None
    evidence_sha256: Optional[str] = None
    snapshot_at: str
    data_window: DataWindow = Field(default_factory=DataWindow)
    data_freshness_days: Optional[int] = None
    sources: List[ProvenanceSource] = Field(default_factory=list)


class ProvenanceChecks(BaseModel):
    guard: GuardOutcome
    # Insights cujo texto exibido veio do código por falta ou recusa do modelo.
    fallback_count: int = 0


class ProvenanceConfidence(BaseModel):
    """Resumo; a confiança de cada insight continua no próprio insight."""

    min: Optional[float] = None
    max: Optional[float] = None
    basis: str = "evidence_count_sources_freshness"


class GenerationProvenance(BaseModel):
    version: int = PROVENANCE_VERSION
    generation_id: str
    kind: str
    prompt: PromptRef
    model: ModelRef
    inputs: ProvenanceInputs
    checks: ProvenanceChecks
    confidence: Optional[ProvenanceConfidence] = None
    output_sha256: str
    # Ids do que a geração produziu (ex.: ids dos insights).
    subject_ids: List[str] = Field(default_factory=list)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def data_window_of(sources: Iterable[ProvenanceSource]) -> DataWindow:
    dates = sorted(s.as_of for s in sources if s.as_of)
    if not dates:
        return DataWindow()
    return DataWindow(**{"from": dates[0], "to": dates[-1]})


def served_by_of(provider: Any) -> Dict[str, Optional[str]]:
    """Quem respondeu, para qualquer provider (também os que não implementam `served_by`)."""
    describe = getattr(provider, "served_by", None)
    if callable(describe):
        return describe()
    return {
        "provider": getattr(provider, "provider_name", None),
        "model": getattr(provider, "_model", None),
    }


def build_generation_provenance(
    *,
    kind: str,
    prompt_id: str,
    prompt_version: str,
    prompt_text: str,
    provider: Any,
    portfolio_snapshot: Any = None,
    evidence: Any = None,
    sources: Optional[List[ProvenanceSource]] = None,
    data_freshness_days: Optional[int] = None,
    guard: GuardOutcome = "unchecked",
    fallback_count: int = 0,
    confidence_values: Optional[List[float]] = None,
    output: Any = None,
    subject_ids: Optional[List[str]] = None,
) -> GenerationProvenance:
    """
    Monta o bloco de uma chamada ao LLM. Recebe os objetos de verdade (prompt,
    carteira, evidência, resposta) e guarda só o hash de cada um.
    """
    sources = sources or []
    served = served_by_of(provider) if guard != "not_called" else {}
    values = confidence_values or []
    return GenerationProvenance(
        generation_id=new_generation_id(),
        kind=kind,
        prompt=PromptRef(
            id=prompt_id, version=prompt_version, sha256=sha256_text(prompt_text)
        ),
        model=ModelRef(
            requested=os.getenv("LLM_PROVIDER", "gemini").lower().strip(),
            served_provider=served.get("provider"),
            model=served.get("model"),
        ),
        inputs=ProvenanceInputs(
            portfolio_snapshot_sha256=(
                sha256_json(portfolio_snapshot)
                if portfolio_snapshot is not None
                else None
            ),
            evidence_sha256=sha256_json(evidence) if evidence is not None else None,
            snapshot_at=utc_now_iso(),
            data_window=data_window_of(sources),
            data_freshness_days=data_freshness_days,
            sources=sources,
        ),
        checks=ProvenanceChecks(guard=guard, fallback_count=fallback_count),
        confidence=(
            ProvenanceConfidence(min=min(values), max=max(values)) if values else None
        ),
        output_sha256=sha256_json(output),
        subject_ids=subject_ids or [],
    )
