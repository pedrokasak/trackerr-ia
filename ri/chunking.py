"""
Quebra do texto de um documento de RI em chunks POR PAGINA (TRA-264).

A pagina vem dos marcadores que o pdf-parse insere no fim de cada pagina
("-- 3 of 10 --", ver ri/fidelity.py). Chunk nunca atravessa pagina: a
citacao mostra UMA pagina, e ela tem de ser a do trecho.

Funcoes puras, sem banco e sem embedding.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ri.fidelity import PAGE_MARKER

# ~375 tokens: um paragrafo longo ou uma tabela pequena inteira. Menor que
# isso, o trecho perde o contexto; maior, a busca mistura assuntos.
CHUNK_CHARS = 1_500
# Sobreposicao entre chunks da mesma pagina: a frase cortada no fim de um
# chunk aparece inteira no seguinte.
CHUNK_OVERLAP = 200
# Teto por documento (~600 mil caracteres): e o limite de custo de embedding
# de um formulario enorme. O inicio do documento entra; o resto fica de fora.
MAX_CHUNKS = 400
# Pedaco menor que isso e numero de pagina, cabecalho ou rodape solto.
MIN_CHUNK_CHARS = 40


@dataclass(frozen=True)
class RiTextChunk:
    page: Optional[int]
    index: int
    text: str


def split_pages(content: str) -> List[Tuple[Optional[int], str]]:
    """
    (pagina, texto) na ordem do documento, com espacos colapsados.

    O texto antes do marcador "-- k of N --" e a pagina k. Depois do ultimo
    marcador vem a pagina seguinte, se ela existir: o server corta
    documentos longos, e o fim do texto nem sempre e o fim do PDF. Texto sem
    marcador nenhum vira uma pagina desconhecida (None).
    """
    text = content or ""
    pages: List[Tuple[Optional[int], str]] = []
    last_end = 0
    last_match = None
    for match in PAGE_MARKER.finditer(text):
        pages.append((int(match.group(1)), text[last_end : match.start()]))
        last_end = match.end()
        last_match = match

    tail = text[last_end:]
    if last_match is None:
        pages.append((None, tail))
    else:
        page, total = int(last_match.group(1)), int(last_match.group(2))
        pages.append((page + 1 if page + 1 <= total else None, tail))

    return [(page, " ".join(body.split())) for page, body in pages if body.strip()]


def chunk_document(content: str) -> List[RiTextChunk]:
    chunks: List[RiTextChunk] = []
    for page, body in split_pages(content):
        for piece in _split_body(body):
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append(RiTextChunk(page=page, index=len(chunks), text=piece))
            if len(chunks) == MAX_CHUNKS:
                return chunks
    return chunks


def _split_body(body: str) -> List[str]:
    """Pedacos de ate CHUNK_CHARS, cortados em espaco, com sobreposicao."""
    if len(body) <= CHUNK_CHARS:
        return [body]

    pieces: List[str] = []
    start = 0
    while start < len(body):
        end = min(len(body), start + CHUNK_CHARS)
        if end < len(body):
            # Corta no ultimo espaco da segunda metade: nunca no meio de uma
            # palavra ou de um numero ("R$ 1.2|34 milhoes").
            space = body.rfind(" ", start + CHUNK_CHARS // 2, end)
            if space > start:
                end = space
        pieces.append(body[start:end].strip())
        if end >= len(body):
            break
        # O proximo comeca CHUNK_OVERLAP antes do corte, numa fronteira de
        # palavra, e sempre avanca.
        next_start = body.find(" ", max(end - CHUNK_OVERLAP, start + 1), end)
        start = next_start + 1 if next_start != -1 else end
    return pieces
