"""
Fidelidade do resumo de RI ao documento-fonte (TRA-239).

Funcoes puras, sem LLM: decidem se o que o modelo escreveu EXISTE no texto
que ele recebeu. Tres perguntas:

1. O trecho citado como evidencia esta no documento? Comparacao por
   "esqueleto" (so letras e digitos, sem acento, minusculo): sobrevive a
   quebra de linha, hifenizacao e espacamento estranho da extracao de PDF,
   mas nao a parafrase.
2. Os numeros citados aparecem na referencia? Comparacao por VALOR, nao por
   grafia: "2,0%" e "2%" batem; "8%" nao bate com "8,1%". Cada numero e lido
   em pt-BR e en-US (ha releases em ingles), e basta uma leitura coincidir.
3. Codigos de periodo (2T26, 1S26, 9M26, 2Q26) e tickers (PETR4) citados
   existem na referencia? Um "1T26" inventado num resumo do 2T26 e erro
   factual mesmo sem numero solto nenhum.

A pagina de um trecho vem dos marcadores que o pdf-parse insere ao fim de
cada pagina ("-- 3 of 10 --"). E calculada aqui, nunca pedida ao modelo:
numero de pagina que o LLM informa e mais uma coisa que ele pode inventar.

Limites conhecidos (deliberados): numero por extenso ("dois bilhoes") nao e
checado; conversao de escala ("R$ 1.200 milhoes" -> "R$ 1,2 bilhao") e
tratada como numero sem suporte — o prompt pede a mesma grafia do documento.
"""

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import FrozenSet, Iterable, List, Optional, Tuple

# Marcador de fim de pagina do pdf-parse v2 (pageJoiner padrao), ja com os
# espacos colapsados pelo server: "... fim da pagina 3 -- 3 of 10 -- inicio da 4".
PAGE_MARKER = re.compile(r"--\s*(\d+)\s+of\s+(\d+)\s*--")

NUMBER_TOKEN = re.compile(r"\d+(?:[.,]\d+)*")
PT_BR_NUMBER = re.compile(r"^\d{1,3}(?:\.\d{3})+(?:,\d+)?$|^\d+(?:,\d+)?$")
EN_US_NUMBER = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$|^\d+(?:\.\d+)?$")

# Sufixos que, colados ao numero, ainda fazem dele uma quantidade:
# "R$ 2,5bi", "1,8x", "150bps". Qualquer outra letra colada ("2T26", "3º",
# "5G") indica codigo, nao quantidade.
UNIT_SUFFIXES = frozenset(
    {"bi", "mi", "mm", "mil", "tri", "k", "x", "pp", "bp", "bps",
     "bilhao", "bilhoes", "milhao", "milhoes"}
)

PERIOD_CODE = re.compile(r"(?<![0-9A-Za-z])([1-4])([TQ])(?:20)?(\d{2})(?![0-9A-Za-z])", re.IGNORECASE)
HALF_OR_YTD_CODE = re.compile(
    r"(?<![0-9A-Za-z])(1S|2S|[1-9]M|1[0-2]M)(?:20)?(\d{2})(?![0-9A-Za-z])", re.IGNORECASE
)
TICKER = re.compile(r"(?<![0-9A-Za-z])([A-Z]{4}\d{1,2})(?![0-9A-Za-z])")

# Trecho mais curto que isso nao prova nada: "receita cresceu" casa em
# qualquer release.
MIN_EXCERPT_SKELETON = 16


@dataclass(frozen=True)
class SourceIndex:
    text: str
    skeleton: str
    # posicao no esqueleto -> posicao no texto original
    positions: Tuple[int, ...]
    # (posicao do marcador no texto original, numero da pagina que ele fecha, total)
    markers: Tuple[Tuple[int, int, int], ...]
    numbers: FrozenSet[Decimal]
    identifiers: FrozenSet[str]


def _skeleton_chars(text: str, skip: Iterable[Tuple[int, int]] = ()):
    spans = sorted(skip)
    span_index = 0
    for i, ch in enumerate(text):
        while span_index < len(spans) and spans[span_index][1] <= i:
            span_index += 1
        if span_index < len(spans) and spans[span_index][0] <= i < spans[span_index][1]:
            continue
        for decomposed in unicodedata.normalize("NFKD", ch):
            if unicodedata.combining(decomposed):
                continue
            lowered = decomposed.lower()
            if lowered.isascii() and lowered.isalnum():
                yield lowered, i


def _skeleton(text: str) -> str:
    return "".join(char for char, _ in _skeleton_chars(text))


def _candidates(token: str) -> FrozenSet[Decimal]:
    values = set()
    try:
        if PT_BR_NUMBER.match(token):
            values.add(Decimal(token.replace(".", "").replace(",", ".")))
        if EN_US_NUMBER.match(token):
            values.add(Decimal(token.replace(",", "")))
    except InvalidOperation:
        pass
    return frozenset(values)


def _letter_run(text: str, start: int) -> str:
    end = start
    while end < len(text) and text[end].isalpha():
        end += 1
    return _skeleton(text[start:end])


def number_values(text: str) -> List[Tuple[str, FrozenSet[Decimal]]]:
    """Numeros do texto como (grafia original, valores possiveis)."""
    text = text or ""
    found: List[Tuple[str, FrozenSet[Decimal]]] = []
    for match in NUMBER_TOKEN.finditer(text):
        start, end = match.span()
        if start > 0 and text[start - 1].isalnum():
            continue
        if end < len(text) and text[end].isalpha():
            if _letter_run(text, end) not in UNIT_SUFFIXES:
                continue
        candidates = _candidates(match.group())
        if candidates:
            found.append((match.group(), candidates))
    return found


def number_set(text: str) -> FrozenSet[Decimal]:
    """Todos os valores numericos do texto, ignorando marcadores de pagina."""
    return frozenset(
        value
        for _, candidates in number_values(strip_page_markers(text or ""))
        for value in candidates
    )


def strip_page_markers(text: str) -> str:
    return " ".join(PAGE_MARKER.sub(" ", text).split())


def unsupported_numbers(claim: str, allowed: FrozenSet[Decimal]) -> List[str]:
    """Numeros do `claim` que nao existem em `allowed`, na grafia original."""
    return [raw for raw, candidates in number_values(claim) if not (candidates & allowed)]


def extract_identifiers(text: str) -> set:
    text = text or ""
    found = set()
    for quarter, letter, year in PERIOD_CODE.findall(text):
        found.add(f"{quarter}{letter.upper()}{year}")
    for prefix, year in HALF_OR_YTD_CODE.findall(text):
        found.add(f"{prefix.upper()}{year}")
    found.update(TICKER.findall(text))
    return found


def unsupported_identifiers(claim: str, allowed: FrozenSet[str]) -> List[str]:
    return sorted(identifier for identifier in extract_identifiers(claim) if identifier not in allowed)


def build_source_index(text: str) -> SourceIndex:
    text = text or ""
    markers = tuple(
        (match.start(), int(match.group(1)), int(match.group(2)))
        for match in PAGE_MARKER.finditer(text)
    )
    marker_spans = [match.span() for match in PAGE_MARKER.finditer(text)]
    chars = list(_skeleton_chars(text, marker_spans))
    return SourceIndex(
        text=text,
        skeleton="".join(char for char, _ in chars),
        positions=tuple(position for _, position in chars),
        markers=markers,
        numbers=number_set(text),
        identifiers=frozenset(extract_identifiers(strip_page_markers(text))),
    )


def locate_excerpt(index: SourceIndex, excerpt: str) -> Optional[int]:
    """Posicao do trecho no texto original, ou None se nao estiver la."""
    span = excerpt_span(index, excerpt)
    return span[0] if span else None


def excerpt_span(index: SourceIndex, excerpt: str) -> Optional[Tuple[int, int]]:
    """Inicio e fim do trecho no texto original — o que o documento diz de fato."""
    needle = _skeleton(excerpt or "")
    if len(needle) < MIN_EXCERPT_SKELETON:
        return None
    found = index.skeleton.find(needle)
    if found < 0:
        return None
    return index.positions[found], index.positions[found + len(needle) - 1] + 1


def page_at(index: SourceIndex, position: Optional[int]) -> Optional[int]:
    if position is None or not index.markers:
        return None
    for marker_start, page, _total in index.markers:
        if marker_start > position:
            return page
    # Depois do ultimo marcador: pagina seguinte, se ela existir. O server
    # corta documentos longos, entao o ultimo marcador nem sempre e o do fim.
    _, last_page, total = index.markers[-1]
    return last_page + 1 if last_page + 1 <= total else None
