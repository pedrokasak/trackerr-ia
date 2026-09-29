"""Quebra do documento de RI em chunks por pagina (TRA-264)."""

from ri.chunking import (
    CHUNK_CHARS,
    MAX_CHUNKS,
    chunk_document,
    split_pages,
)


def test_texto_antes_do_marcador_e_a_pagina_dele():
    content = "Capa do release -- 1 of 3 -- Receita cresceu 12% -- 2 of 3 -- Dividendos aprovados"

    assert split_pages(content) == [
        (1, "Capa do release"),
        (2, "Receita cresceu 12%"),
        (3, "Dividendos aprovados"),
    ]


def test_depois_do_ultimo_marcador_vem_a_pagina_seguinte_se_existir():
    # O server corta documento longo: o fim do texto nao e o fim do PDF.
    assert split_pages("A -- 1 of 10 -- B")[-1] == (2, "B")
    # Ja no fim do PDF: nao inventa pagina 4 de 3.
    assert split_pages("A -- 3 of 3 -- rodape")[-1] == (None, "rodape")


def test_texto_sem_marcador_vira_pagina_desconhecida():
    assert split_pages("Fato relevante sem marcadores") == [
        (None, "Fato relevante sem marcadores")
    ]


def test_chunk_nunca_atravessa_pagina():
    page_one = "Receita líquida de R$ 1.234 milhões no trimestre. " * 5
    page_two = "Dividendos de R$ 0,50 por ação aprovados pelo conselho. " * 5
    chunks = chunk_document(f"{page_one} -- 1 of 2 -- {page_two}")

    assert [chunk.page for chunk in chunks] == [1, 2]
    assert "Dividendos" not in chunks[0].text
    assert [chunk.index for chunk in chunks] == [0, 1]


def test_pagina_longa_vira_varios_chunks_com_sobreposicao_sem_cortar_palavra():
    words = [f"palavra{i}" for i in range(600)]
    chunks = chunk_document(" ".join(words) + " -- 1 of 1 --")

    assert len(chunks) > 1
    assert all(len(chunk.text) <= CHUNK_CHARS for chunk in chunks)
    # Cada chunk comeca e termina em palavra inteira.
    for chunk in chunks:
        assert chunk.text.split()[0] in words
        assert chunk.text.split()[-1] in words
    # Sobreposicao: o fim de um chunk reaparece no inicio do seguinte.
    assert chunks[0].text.split()[-1] in chunks[1].text.split()


def test_descarta_pedaco_curto_demais():
    chunks = chunk_document("12 -- 1 of 2 -- Texto de verdade da segunda página do release.")

    assert [chunk.page for chunk in chunks] == [2]


def test_teto_de_chunks_por_documento():
    page = "Receita cresceu e a margem melhorou no trimestre. " * 30
    content = " ".join(f"{page} -- {i} of 999 --" for i in range(1, 900))

    assert len(chunk_document(content)) == MAX_CHUNKS
