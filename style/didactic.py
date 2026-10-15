"""
Guia de estilo didático dos textos gerados por IA (docs/requisitos/ia-didatica.md,
RF-6). Um bloco de regras de redação, igual em todo prompt que escreve para o
usuário, com o nível da pessoa como parâmetro.

Só muda a forma de explicar: as regras de cada prompt (números permitidos,
tamanho, proibição de recomendar) continuam valendo e vêm antes deste bloco.
"""

from typing import Literal, Optional

DidacticLevel = Literal["iniciante", "intermediario", "avancado"]

# O server fala os dois vocabulários: o do perfil (beginner...) e o da tela.
_ALIASES = {
    "iniciante": "iniciante",
    "beginner": "iniciante",
    "intermediario": "intermediario",
    "intermediário": "intermediario",
    "intermediate": "intermediario",
    "avancado": "avancado",
    "avançado": "avancado",
    "experienced": "avancado",
    "advanced": "avancado",
}

_BY_LEVEL = {
    "iniciante": (
        "- A pessoa está começando: nada de jargão. Se um termo técnico for "
        "inevitável, explique-o na mesma frase com palavras do dia a dia.\n"
        "- Quando ajudar, use uma comparação simples do cotidiano e traduza o "
        "número para reais, se o valor em reais estiver nos dados."
    ),
    "intermediario": (
        "- A pessoa conhece o básico: use o termo técnico com uma definição "
        "curta entre parênteses na primeira vez (ex.: \"yield (proventos de 12 "
        "meses sobre o preço)\").\n"
        "- Compare o número com a referência que estiver nos dados (índice, "
        "limite da política, meta)."
    ),
    "avancado": (
        "- A pessoa é experiente: linguagem técnica e objetiva, sem analogias "
        "nem definições de termos comuns.\n"
        "- Priorize métricas, premissas e a relação entre os números."
    ),
}


def normalize_level(value: Optional[str]) -> DidacticLevel:
    """Nível conhecido ou intermediário (o meio-termo, como no server)."""
    if not value:
        return "intermediario"
    return _ALIASES.get(str(value).strip().lower(), "intermediario")  # type: ignore[return-value]


def didactic_rules(level: Optional[str]) -> str:
    """Bloco de redação didática para colar no prompt, depois das regras fixas."""
    resolved = normalize_level(level)
    return (
        "COMO EXPLICAR (estilo didático do Trackerr):\n"
        "- Comece pela resposta direta, com o número principal e a referência dele.\n"
        "- Diga o que o número significa e por que importa para a pessoa, de "
        "preferência com a consequência concreta.\n"
        "- Mostre consequências e opções, nunca uma ordem; a decisão é da pessoa.\n"
        f"{_BY_LEVEL[resolved]}"
    )
