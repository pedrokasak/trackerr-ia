"""
Ajustes no texto transcrito antes de ele virar pergunta para o orquestrador.

- Tickers: o Whisper ouve "Petri4", "Peter 4" ou "YouTube 4" onde a pessoa
  disse PETR4 ou ITUB4. O `prompt` com os tickers da carteira resolve a
  maioria (6 de 6 no spike com gravações reais); isto pega o resto,
  comparando com os tickers que o server enviou.
- Frase inventada: em trecho de silêncio o Whisper às vezes devolve
  "Obrigado." no começo. Uma frase de cortesia sozinha no início sai.
"""

import re
from difflib import SequenceMatcher
from typing import Iterable

# Letras (com acento) e, colado ou separado por espaço, o número do ticker.
_CANDIDATE = re.compile(r"\b([A-Za-zÀ-ÿ]{3,10})\s?(\d{1,2})\b")
_TICKER = re.compile(r"^([A-Z]{4})(\d{1,2})$")
_MIN_SIMILARITY = 0.5
# Palavra curta parecida com a raiz por acaso ("mês 3" e MRVE3) precisa de
# semelhança maior.
_MIN_SIMILARITY_SHORT = 0.75
# "quanto vale 3 cotas": o número é quantidade, não parte de ticker.
_QUANTITY_AFTER = re.compile(
    r"\s*(a[cç][oõ]es|cotas?|mil|milh[oõ]es|reais|anos?|meses|dias?|vezes|por\s?cento|%)(?!\w)",
    re.IGNORECASE,
)
_COURTESY = re.compile(
    r"^\s*(obrigad[oa]|tchau|valeu|legendas? pela comunidade[^.!?]*|amém)\s*[.!?]+(\s+|$)",
    re.IGNORECASE,
)


def _fold(text: str) -> str:
    table = str.maketrans("ÁÀÂÃÉÊÍÓÔÕÚÜÇáàâãéêíóôõúüç", "AAAAEEIOOOUUCaaaaeeiooouuc")
    return text.translate(table).upper()


def normalize_tickers(text: str, known_tickers: Iterable[str]) -> str:
    """
    Troca "palavra + número" pelo ticker conhecido com o mesmo número e a
    raiz mais parecida. Sem candidato único acima do limiar, deixa como está:
    é melhor não corrigir do que corrigir para o ticker errado.
    """
    by_number: dict[str, list[tuple[str, str]]] = {}
    for ticker in known_tickers:
        match = _TICKER.match(ticker.strip().upper())
        if match:
            by_number.setdefault(match.group(2), []).append((match.group(1), ticker.strip().upper()))
    if not by_number:
        return text

    def replace(match: re.Match) -> str:
        word, number = _fold(match.group(1)), match.group(2)
        if _QUANTITY_AFTER.match(text, match.end()):
            return match.group(0)
        candidates = by_number.get(number, [])
        scored = sorted(
            ((SequenceMatcher(None, word, root).ratio(), ticker) for root, ticker in candidates),
            reverse=True,
        )
        threshold = _MIN_SIMILARITY_SHORT if len(word) <= 3 else _MIN_SIMILARITY
        if not scored or scored[0][0] < threshold:
            return match.group(0)
        if len(scored) > 1 and scored[1][0] == scored[0][0]:
            return match.group(0)
        return scored[0][1]

    return _CANDIDATE.sub(replace, text)


def strip_courtesy_prefix(text: str) -> str:
    """Remove uma frase de cortesia isolada no início ("Obrigado. A Vale…")."""
    return _COURTESY.sub("", text, count=1)


def clean_transcript(text: str, known_tickers: Iterable[str]) -> str:
    return normalize_tickers(strip_courtesy_prefix(text), known_tickers).strip()
