# ETFs — RASCUNHO (revisar antes de publicar · TRA-36)

> Rascunho gerado por IA. Confirmar vigência, números e fonte com
> profissional habilitado (CRC). Ver `README.md`.
>
> **Atualizado em 2026-10-28** (exercício 2027). Ver a linha *Conferência
> interna* de cada chunk.

### FISC-020 — Ganho de capital em ETF de ações
- **chunk_id:** `fiscal:etf:acoes-ganho-capital`
- **Regra (rascunho):** O ganho de capital na venda de cotas de ETF de ações é
  tributado à alíquota de **15%** em operações comuns, **sem** a isenção de
  R$ 20.000/mês aplicável a ações individuais. Day trade com ETF segue a
  alíquota de day trade (20%). O prejuízo na venda de ETF compensa ganho de
  operações comuns, a mesma categoria das ações à vista.
- **Fonte oficial (a confirmar):** Lei nº 11.033/2004, art. 2º, II ("demais
  hipóteses"); IN RFB nº 1.585/2015.
- **Conferência interna (2026-10-25):** a alíquota geral de bolsa foi lida na
  lei. **Enquadrar o ETF nessa regra é entendimento da Receita**, de norma
  conhecida e **não relido na fonte**.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar 15%, que **a isenção de R$ 20.000 não vale
  para ETF de ações** e que o prejuízo de ETF e o de ações à vista compartilham
  o mesmo saldo de operações comuns.

### FISC-021 — ETF de renda fixa
- **chunk_id:** `fiscal:etf:renda-fixa`
- **Regra (rascunho):** ETFs de renda fixa têm tributação própria, com
  alíquotas que **variam conforme o prazo médio de repactuação da carteira do
  fundo** (a confirmar), diferente da tabela regressiva usual de renda fixa. A
  retenção costuma ocorrer na fonte, no resgate/venda.
- **Fonte oficial (a confirmar):** Lei nº 13.043/2014; IN RFB aplicável a ETF
  de renda fixa.
- **Conferência interna:** não verificado; fora do escopo do cálculo do
  produto.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar as faixas de alíquota por prazo médio e a
  forma de retenção. Este ponto é tecnicamente sensível — verificar com
  cuidado antes de aprovar.

### FISC-022 — Isenção de R$ 20.000 não se aplica a ETF
- **chunk_id:** `fiscal:etf:sem-isencao-20k`
- **Regra (rascunho):** A isenção mensal de R$ 20.000 em vendas é específica de
  **ações à vista**. Não se estende a ETFs (nem de ações, nem de renda fixa)
  nem a FIIs.
- **Fonte oficial (a confirmar):** Lei nº 11.033/2004, art. 3º, I; IN RFB nº
  1.585/2015.
- **Conferência interna (2026-10-25):** o texto da isenção fala em "conjunto de
  ações".
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar a delimitação da isenção a ações à vista.

### FISC-023 — Como distinguir ETF, ação e unit (tickers terminados em 11)
- **chunk_id:** `fiscal:etf:ticker-11-etf-ou-unit`
- **Regra (rascunho):** O final "11" no código do ativo **não diz sozinho** o
  que ele é. Pode ser um **ETF** (por exemplo BOVA11 e IVVB11), um **FII** ou
  uma **unit**, que é um pacote de ações negociado junto (por exemplo SANB11).
  Para o imposto, o ETF segue a regra do ETF (15%, sem a isenção de R$ 20.000);
  a unit é tratada como ação (com a isenção, se o total vendido no mês for de
  até R$ 20.000); o FII segue a regra de FII (20%). Classificar errado muda o
  imposto estimado.
- **Fonte oficial (a confirmar):** Lei nº 11.033/2004; normas da B3 sobre os
  tipos de ativo.
- **Conferência interna (2026-10-28):** é a regra que o produto passou a
  aplicar (o catálogo de ETFs e units conhecidos é conservador; ticker
  desconhecido terminado em 11 não é promovido a ETF). Que a **unit seja tratada
  como ação** para a isenção de R$ 20.000 é **entendimento a confirmar**.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar o tratamento da unit para a isenção de
  R$ 20.000 e para a compensação de prejuízo.
