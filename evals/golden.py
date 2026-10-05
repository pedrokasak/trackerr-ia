"""
Dataset dourado da avaliação (TRA-242): perguntas e contextos sintéticos,
sem dado de usuário, com a resposta que o gerador deu e a nota do juiz.

    python -m evals.golden record   # gera respostas e notas (usa LLM)
    python -m evals.golden check    # confere o gravado (sem LLM; é o do CI)

O CI roda o `check` (tests/test_golden_eval_gate.py): falha se um prompt
mudou desde a gravação ou se a fidelidade ficou abaixo do limiar. Trocar o
modelo no ambiente (OPENROUTER_MODEL, LLM_PROVIDER) não muda o código: rode
o `record` antes de trocar e confira o resultado.
"""

import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from evals.checks import check_rag_answer
from evals.fingerprint import prompt_fingerprint
from evals.judge import RUBRIC_VERSION, LlmJudge, build_judge_provider

GOLDEN_PATH = Path(__file__).parent / "golden" / "rag_golden.json"


@dataclass(frozen=True)
class _Chunk:
    content: str


async def _with_retry(call, attempts: int = 4, first_wait: float = 5.0):
    """Gravação manual: limite de taxa de plano gratuito não pode abortar tudo."""
    wait = first_wait
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except Exception as error:  # noqa: BLE001 - o motivo vai no print
            if attempt == attempts:
                raise
            print(f"  tentativa {attempt} falhou ({type(error).__name__}); nova em {wait:.0f}s")
            await asyncio.sleep(wait)
            wait *= 2


def load_golden(path: Path = GOLDEN_PATH) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_golden(golden: Dict[str, Any]) -> List[str]:
    """Problemas do dataset gravado; lista vazia quando passa."""
    problems: List[str] = []
    current = prompt_fingerprint()
    if golden.get("fingerprint") != current:
        problems.append(
            f"prompt mudou desde a gravação ({golden.get('fingerprint')} -> {current}): "
            "rode `python -m evals.golden record`"
        )
    if golden.get("rubric_version") != RUBRIC_VERSION:
        problems.append("rubrica mudou desde a gravação: rode `python -m evals.golden record`")

    threshold = float(golden.get("thresholds", {}).get("fidelity", 0.8))
    fidelities = []
    for case in golden.get("cases", []):
        answer = case.get("answer") or ""
        if not answer:
            problems.append(f"{case['id']}: sem resposta gravada")
            continue
        checks = check_rag_answer(case["question"], case["context"], answer)
        if checks.unsupported_numbers:
            problems.append(f"{case['id']}: números sem base: {checks.unsupported_numbers}")
        if checks.recommendation_language:
            problems.append(f"{case['id']}: linguagem de recomendação")
        if not checks.disclaimer_present:
            problems.append(f"{case['id']}: sem disclaimer")
        judge = case.get("judge") or {}
        if judge.get("recommendation_language"):
            problems.append(f"{case['id']}: o juiz viu recomendação")
        if judge.get("numeric_hallucination"):
            problems.append(f"{case['id']}: o juiz viu número inventado")
        if isinstance(judge.get("fidelity"), (int, float)):
            fidelities.append(float(judge["fidelity"]))
        else:
            problems.append(f"{case['id']}: sem nota de fidelidade do juiz")

    if fidelities:
        mean = sum(fidelities) / len(fidelities)
        if mean < threshold:
            problems.append(f"fidelidade média {mean:.3f} abaixo do limiar {threshold}")
    return problems


async def record(path: Path = GOLDEN_PATH) -> Dict[str, Any]:
    """Grava resposta e nota de cada caso com o gerador e o juiz configurados."""
    from benchmark.providers.factory import LLMFactory
    from rag.query_service import DISCLAIMER, RagQueryService

    golden = load_golden(path)
    generator = LLMFactory.get_provider()
    judge_provider = build_judge_provider()
    if judge_provider is None:
        raise SystemExit("Sem juiz: configure EVAL_JUDGE_PROVIDER ou a chave de outro provider.")
    judge = LlmJudge(judge_provider)

    for case in golden["cases"]:
        chunks = [_Chunk(line) for line in case["context"].splitlines() if line.strip()]
        # Mesmo prompt e mesmo formato final do RAG em produção.
        prompt = RagQueryService._build_prompt(None, case["question"], chunks)
        raw = await _with_retry(lambda: generator.analyze(prompt))
        body = str(raw.get("answer") or raw.get("raw_response") or "").strip()
        case["answer"] = f"{body}\n\n{DISCLAIMER}"
        verdict = await _with_retry(
            lambda: judge.judge("rag", case["question"], case["answer"], case["context"])
        )
        print(f"{case['id']}: gravado")
        case["judge"] = (
            {
                "fidelity": verdict.fidelity,
                "numeric_hallucination": verdict.numeric_hallucination,
                "recommendation_language": verdict.recommendation_language,
                "usefulness": verdict.usefulness,
                "level_fit": verdict.level_fit,
            }
            if verdict
            else None
        )

    golden["fingerprint"] = prompt_fingerprint()
    golden["rubric_version"] = RUBRIC_VERSION
    golden["recorded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    golden["generator"] = generator.provider_name
    golden["judge"] = judge.provider_name
    path.write_text(json.dumps(golden, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return golden


def main(argv: List[str]) -> int:
    command = argv[1] if len(argv) > 1 else "check"
    if command == "record":
        load_dotenv()
        asyncio.run(record())
    problems = check_golden(load_golden())
    for problem in problems:
        print(f"FALHOU: {problem}")
    if not problems:
        print("Dataset dourado OK.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
