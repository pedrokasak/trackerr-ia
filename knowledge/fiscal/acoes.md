# Ações — RASCUNHO (revisar antes de publicar · TRA-36)

> Rascunho gerado por IA. Cada chunk precisa de confirmação de vigência,
> números e fonte por profissional habilitado (CRC). Ver `README.md`.
>
> **Atualizado em 2026-10-28** para as regras do exercício 2027
> (ano-calendário 2026). A linha *Conferência interna* de cada chunk diz o que
> já foi lido na fonte e o que ainda é só de imprensa ou corretora. Ela não
> substitui a sua confirmação.

### FISC-001 — Ganho de capital em ações (operação comum / swing trade)
- **chunk_id:** `fiscal:acoes:ganho-capital-comum`
- **Regra (rascunho):** No mercado à vista de ações, o ganho de capital em
  operações comuns (não day trade) é tributado à alíquota de **15%** sobre o
  lucro líquido do mês. O imposto é apurado e pago pelo próprio investidor,
  pessoa física, via DARF mensal.
- **Fonte oficial (a confirmar):** Lei nº 11.033/2004, art. 2º, II; IN RFB nº
  1.585/2015.
- **Conferência interna (2026-10-25):** alíquota lida no texto da lei em
  planalto.gov.br. A MP nº 1.303/2025, que previa alíquota única de 17,5% ou
  18%, **perdeu a validade em 08/10/2025**; o texto da Lei 11.033 no Planalto
  marca os dispositivos que ela alterou como "Vigência encerrada". Qualquer
  fonte que cite 17,5% ou 18% como regra vigente está desatualizada.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar que não há outra alteração em vigor para
  o ano-calendário 2026.

### FISC-002 — Isenção de vendas até R$ 20.000 no mês (ações à vista)
- **chunk_id:** `fiscal:acoes:isencao-20k`
- **Regra (rascunho):** O ganho obtido na venda de ações no mercado à vista é
  **isento de imposto de renda quando o total de vendas de ações no mês não
  ultrapassa R$ 20.000**. A isenção vale para o total vendido no mês, somando
  todas as ações, e **não se aplica a day trade**. Passou de R$ 20.000, o ganho
  do mês inteiro é tributado.
- **Fonte oficial (a confirmar):** Lei nº 11.033/2004, art. 2º, II e art. 3º, I.
- **Conferência interna (2026-10-25):** limite e base (valor vendido, não o
  lucro) lidos na lei; o texto diz "para o conjunto de ações".
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar que só vale para ações à vista (não FII,
  não ETF, não day trade). Em mês isento, o prejuízo acumulado continua
  acumulando e o ganho isento não consome prejuízo antigo; confirmar esse
  tratamento.

### FISC-003 — Day trade em ações
- **chunk_id:** `fiscal:acoes:day-trade`
- **Regra (rascunho):** Operações de day trade (compra e venda do mesmo ativo
  no mesmo dia) são tributadas à alíquota de **20%** sobre o ganho líquido.
  **Não há isenção** de R$ 20.000 para day trade.
- **Fonte oficial (a confirmar):** Lei nº 11.033/2004, art. 2º, I; IN RFB nº
  1.585/2015.
- **Conferência interna (2026-10-25):** alíquota lida na lei.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar a definição de day trade adotada pela
  Receita.

### FISC-004 — Imposto retido na fonte ("dedo-duro")
- **chunk_id:** `fiscal:acoes:irrf-dedo-duro`
- **Regra (rascunho):** Há retenção de IR na fonte, chamada informalmente de
  "dedo-duro", com função de informar a operação à Receita: **0,005%** sobre o
  valor da venda em operações comuns e **1%** sobre o ganho em day trade. O
  valor retido pode ser deduzido do imposto devido na apuração.
- **Fonte oficial (a confirmar):** Lei nº 11.033/2004, art. 2º, § 1º, IV
  (0,005%); IN RFB nº 1.585/2015.
- **Conferência interna (2026-10-25):** o 0,005% foi lido na lei. O **1% do day
  trade é só de fonte secundária**: a lei remete a outro texto.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar o 1% e a base (ganho), e a regra de
  compensação do retido.

### FISC-005 — Prazo e forma de pagamento (DARF)
- **chunk_id:** `fiscal:acoes:darf-prazo`
- **Regra (rascunho):** O imposto sobre ganho de capital em ações é apurado
  **mensalmente** e pago via DARF (código **6015**, a confirmar) até o **último
  dia útil do mês seguinte** ao da apuração. DARF abaixo de R$ 10,00 não é
  emitido na hora: o valor soma ao do mês seguinte.
- **Fonte oficial (a confirmar):** IN RFB nº 1.585/2015.
- **Conferência interna (2026-10-25):** código, prazo e mínimo de R$ 10 são de
  norma conhecida, **não relidos na fonte oficial**.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar o código 6015, o prazo e o mínimo de
  R$ 10,00.

### FISC-006 — Dividendos (regra geral e mudança de 2026)
- **chunk_id:** `fiscal:acoes:dividendos`
- **Regra (rascunho):** Os dividendos continuam, em regra, **isentos** para a
  pessoa física que os recebe. **Desde janeiro de 2026 isso mudou em parte**:
  há retenção de 10% quando uma mesma empresa paga mais de R$ 50.000 no mês à
  mesma pessoa (veja `fiscal:acoes:dividendos-retencao-10`), e os dividendos
  entram na base do imposto mínimo de altas rendas (veja
  `fiscal:apuracao:irpfm`). Os dividendos aparecem na declaração em
  "Rendimentos Isentos e Não Tributáveis".
- **Fonte oficial (a confirmar):** Lei nº 9.249/1995, art. 10; Lei nº
  9.250/1995, art. 6º-A e art. 16-A (incluídos pela Lei nº 15.270/2025).
- **Conferência interna (2026-10-25):** os arts. 6º-A e 16-A foram lidos no
  texto da lei. A isenção geral do art. 10 da Lei 9.249 é de norma conhecida,
  **não relida hoje**.
- **Status:** ⬜ pendente
- **Notas para o revisor:** **prioridade**. Confirmar o alcance da isenção que
  resta depois da Lei 15.270 e a ficha em que cada caso é declarado.

### FISC-007 — Juros sobre capital próprio (JCP)
- **chunk_id:** `fiscal:acoes:jcp`
- **Regra (rascunho):** Os juros sobre capital próprio (JCP) recebidos são
  tributados na fonte à alíquota de **15%**, retidos no pagamento, em
  tributação **exclusiva**. Diferente dos dividendos, o JCP é tributado. Na
  declaração, o JCP vai em "Rendimentos Sujeitos à Tributação Exclusiva/
  Definitiva", não em "Rendimentos Isentos". O valor creditado pela corretora
  costuma ser o **líquido**; o bruto e o imposto retido estão no informe de
  rendimentos.
- **Fonte oficial (a confirmar):** Lei nº 9.249/1995, art. 9º.
- **Conferência interna (2026-10-25):** alíquota de 15% é de norma conhecida,
  consistente com o produto, **não relida na fonte oficial**. Os 17,5%/18% da
  MP 1.303 não estão em vigor (a MP perdeu a validade).
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar os 15% e que são definitivos para a
  pessoa física. Confirmar também que o JCP entra na base do imposto mínimo de
  altas rendas (rendimentos de tributação exclusiva entram, segundo o texto
  lido do art. 16-A).

### FISC-008 — Retenção de 10% sobre dividendos acima de R$ 50.000 no mês
- **chunk_id:** `fiscal:acoes:dividendos-retencao-10`
- **Regra (rascunho):** Desde janeiro de 2026, quando uma **mesma empresa** paga
  a uma **mesma pessoa física**, residente no Brasil, **mais de R$ 50.000 em
  dividendos no mês**, há retenção de **10% sobre o total pago**, e não só
  sobre o que passa de R$ 50.000, sem deduções. Vários pagamentos da mesma
  empresa no mesmo mês se somam. Ficam fora da retenção os lucros apurados até
  2025 cuja distribuição foi aprovada até 31/12/2025, pagos conforme o ato de
  aprovação.
- **Fonte oficial (a confirmar):** Lei nº 9.250/1995, art. 6º-A (incluído pela
  Lei nº 15.270/2025).
- **Conferência interna (2026-10-25):** lido no texto da lei.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar a leitura de que o 10% incide sobre o
  total e não sobre o excedente, e como esse imposto retido é tratado na
  declaração anual.
