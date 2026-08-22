# Ações — RASCUNHO (revisar antes de publicar · TRA-36)

> Rascunho gerado por IA. Cada chunk precisa de confirmação de vigência,
> números e fonte por profissional habilitado (CRC). Ver `README.md`.

### FISC-001 — Ganho de capital em ações (operação comum / swing trade)
- **chunk_id:** `fiscal:acoes:ganho-capital-comum`
- **Regra (rascunho):** No mercado à vista de ações, o ganho de capital em
  operações comuns (não day trade) é tributado à alíquota de **15%** (a
  confirmar) sobre o lucro líquido do mês. O imposto é apurado e pago pelo
  próprio investidor, pessoa física, via DARF mensal.
- **Fonte oficial (a confirmar):** Lei nº 11.033/2004; IN RFB nº 1.585/2015.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar se a alíquota de 15% segue vigente e se
  não foi alterada por reforma da tributação de investimentos.

### FISC-002 — Isenção de vendas até R$ 20.000 no mês (ações à vista)
- **chunk_id:** `fiscal:acoes:isencao-20k`
- **Regra (rascunho):** O ganho obtido na venda de ações no mercado à vista é
  **isento de imposto de renda quando o total de vendas de ações no mês não
  ultrapassa R$ 20.000** (a confirmar). A isenção vale para o total vendido no
  mês, somando todas as ações, e **não se aplica a day trade**.
- **Fonte oficial (a confirmar):** Lei nº 11.033/2004, art. 3º; IN RFB nº
  1.585/2015.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar o limite de R$ 20.000 e que ele é sobre
  o **total de vendas** no mês, não sobre o lucro. Confirmar que só vale para
  ações à vista (não FII, não ETF, não day trade).

### FISC-003 — Day trade em ações
- **chunk_id:** `fiscal:acoes:day-trade`
- **Regra (rascunho):** Operações de day trade (compra e venda do mesmo ativo
  no mesmo dia) são tributadas à alíquota de **20%** (a confirmar) sobre o
  ganho líquido. **Não há isenção** de R$ 20.000 para day trade.
- **Fonte oficial (a confirmar):** Lei nº 11.033/2004; IN RFB nº 1.585/2015.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar alíquota de 20% e a inexistência de
  isenção. Confirmar a definição de day trade adotada pela Receita.

### FISC-004 — Imposto retido na fonte ("dedo-duro")
- **chunk_id:** `fiscal:acoes:irrf-dedo-duro`
- **Regra (rascunho):** Há retenção de IR na fonte, chamada informalmente de
  "dedo-duro", com função de informar a operação à Receita: **0,005%** sobre o
  valor da venda em operações comuns e **1%** sobre o ganho em day trade (a
  confirmar). O valor retido pode ser deduzido do imposto devido na apuração.
- **Fonte oficial (a confirmar):** IN RFB nº 1.585/2015.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar os percentuais (0,005% e 1%) e as bases
  (valor da venda vs. ganho). Confirmar a regra de compensação do retido.

### FISC-005 — Prazo e forma de pagamento (DARF)
- **chunk_id:** `fiscal:acoes:darf-prazo`
- **Regra (rascunho):** O imposto sobre ganho de capital em ações é apurado
  **mensalmente** e pago via DARF (código **6015**, a confirmar) até o **último
  dia útil do mês seguinte** ao da apuração.
- **Fonte oficial (a confirmar):** IN RFB nº 1.585/2015.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar o código de DARF (6015) e o prazo (último
  dia útil do mês seguinte).

### FISC-006 — Dividendos
- **chunk_id:** `fiscal:acoes:dividendos`
- **Regra (rascunho):** Dividendos distribuídos por empresas são, atualmente,
  **isentos de imposto de renda** para a pessoa física que os recebe (a
  confirmar). Esta é uma regra sob discussão recorrente de reforma.
- **Fonte oficial (a confirmar):** Lei nº 9.249/1995, art. 10.
- **Status:** ⬜ pendente
- **Notas para o revisor:** **prioridade** — confirmar se a isenção de
  dividendos na PF continua vigente, dado que é alvo frequente de propostas de
  mudança. Se mudou, este chunk precisa refletir a regra nova.

### FISC-007 — Juros sobre capital próprio (JCP)
- **chunk_id:** `fiscal:acoes:jcp`
- **Regra (rascunho):** Os juros sobre capital próprio (JCP) recebidos são
  tributados na fonte à alíquota de **15%** (a confirmar), retidos no
  pagamento. Diferente dos dividendos, o JCP é tributado.
- **Fonte oficial (a confirmar):** Lei nº 9.249/1995, art. 9º.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar a alíquota de 15% na fonte e que é
  definitiva para a PF.
