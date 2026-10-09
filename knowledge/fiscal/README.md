# Base de conhecimento fiscal — RASCUNHO PARA REVISÃO (TRA-36)

> ## ⚠️ NÃO PUBLICAR SEM REVISÃO PROFISSIONAL
>
> Este é um **rascunho gerado com auxílio de IA**. **Não foi revisado por
> contador ou profissional habilitado** e **não pode ser embedado no RAG nem
> exibido a nenhum usuário** enquanto cada entrada não estiver aprovada por
> um profissional com registro no CRC.
>
> O objetivo deste documento é **reduzir o trabalho do revisor** — dar a
> estrutura e um primeiro esboço do texto de cada regra — e **não** ser fonte
> de verdade fiscal. Nenhuma afirmação aqui deve ser tratada como correta até
> ser confirmada.

## Por que este documento existe

`RagQueryService` já bloqueia, em código, qualquer cálculo definitivo de
imposto (`rag/response_guard.py`) e anexa disclaimer incondicional. O que
falta pra `portfolio_tax` virar um `source_type` real (ver TRA-76) é
**conteúdo curado sobre as regras gerais** — explicação de como o IR incide
sobre cada instrumento, nunca o cálculo do imposto de um usuário específico.

Este arquivo é esse conteúdo em estado de rascunho.

## O que o revisor precisa fazer em cada entrada

Cada regra abaixo é um **chunk** independente (a unidade que o RAG vai
embedar). Para cada uma, o revisor deve:

1. **Confirmar a vigência.** A tributação de investimentos no Brasil muda por
   lei com frequência (a IA que rascunhou isto tem corte de conhecimento e
   **não sabe o que está em vigor hoje**). Confirmar especialmente alíquotas,
   faixas e prazos, e se alguma reforma recente alterou a regra.
2. **Confirmar os números.** Toda alíquota, limite de isenção e código de
   DARF está marcado como *a confirmar*.
3. **Preencher a fonte oficial exata** (lei, IN RFB, artigo). Os campos de
   fonte trazem candidatos, não citações verificadas.
4. **Aprovar ou corrigir o texto** do chunk. O texto é o que o usuário vai
   ler narrado — precisa estar correto e sem induzir a decisão.
5. **Marcar o status** de ⬜ pendente para ✅ aprovado (ou ❌ removido).

## Regra de ouro do conteúdo

- Explica **regra geral**, nunca calcula o imposto de um usuário.
- Nunca recomenda comprar, vender ou "otimizar" tributação — descreve como a
  norma funciona.
- Trata qualquer número que dependa da situação do usuário como **estimativa
  educativa**, nunca valor definitivo.

## Estrutura dos arquivos

| Arquivo | Tema |
|---|---|
| `acoes.md` | Ações à vista, day trade, dividendos, JCP |
| `fiis.md` | Fundos imobiliários |
| `etfs.md` | ETFs de ações e de renda fixa |
| `fundos.md` | Fundos abertos e come-cotas |
| `renda-fixa.md` | Tesouro, CDB, LCI/LCA, debêntures |
| `apuracao.md` | Prejuízo compensável, DARF, prazos, multas |

Cada chunk segue este formato:

```
### FISC-000 — Título curto
- **chunk_id:** fiscal:tema:identificador   (source_type = "fiscal")
- **Regra (rascunho):** texto que o usuário lê
- **Fonte oficial (a confirmar):** candidato de lei/IN/artigo
- **Status:** ⬜ pendente | ✅ aprovado | ❌ removido
- **Notas para o revisor:** o que confere primeiro
```

## Depois da revisão

Quando um conjunto de chunks estiver ✅, o próximo passo técnico (issue
separada) é o produtor que os embeda **uma vez** e os compartilha entre todos
os usuários — dado não-pessoal, ao contrário dos chunks de carteira. O schema
atual de `DocumentChunk` exige `user_id`; conteúdo compartilhado vai precisar
de estratégia própria (sentinel de `user_id` ou tabela separada), decidida na
implementação, não aqui.

---

**Versão do rascunho:** 0.1 (2026-08-21) · **Status geral:** ⬜ nenhuma
entrada aprovada
