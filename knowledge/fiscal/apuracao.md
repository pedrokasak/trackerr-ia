# Apuração, prejuízo e DARF — RASCUNHO (revisar · TRA-36)

> Rascunho gerado por IA. Confirmar vigência, números e fonte com
> profissional habilitado (CRC). Ver `README.md`.

### FISC-050 — Prejuízo compensável por modalidade
- **chunk_id:** `fiscal:apuracao:prejuizo-por-modalidade`
- **Regra (rascunho):** Prejuízos em renda variável podem ser compensados com
  **ganhos futuros da mesma modalidade** (a confirmar): prejuízo de operações
  comuns compensa ganhos de operações comuns; prejuízo de day trade compensa
  apenas ganhos de day trade. A compensação não tem prazo de validade, mas
  precisa de controle contínuo mês a mês.
- **Fonte oficial (a confirmar):** IN RFB nº 1.585/2015.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar a separação por modalidade (comum vs.
  day trade) e que o prejuízo acumulado não expira.

### FISC-051 — Controle mensal e continuidade
- **chunk_id:** `fiscal:apuracao:controle-mensal`
- **Regra (rascunho):** A apuração de imposto sobre renda variável é
  **mensal**. O prejuízo de um mês é levado para abater ganhos dos meses
  seguintes. Quem não vende nada em um mês não tem imposto a pagar naquele mês,
  mas o controle do prejuízo acumulado deve continuar.
- **Fonte oficial (a confirmar):** IN RFB nº 1.585/2015.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar a periodicidade mensal e o carregamento
  do prejuízo.

### FISC-052 — DARF: código, prazo e valor mínimo
- **chunk_id:** `fiscal:apuracao:darf`
- **Regra (rascunho):** O imposto sobre ganho em renda variável é recolhido
  pelo próprio investidor via **DARF**, código **6015** (a confirmar), até o
  **último dia útil do mês seguinte** ao da apuração. Há um **valor mínimo**
  para emissão de DARF (a confirmar); abaixo dele, o imposto é acumulado para
  o mês seguinte.
- **Fonte oficial (a confirmar):** IN RFB nº 1.585/2015; regra geral de DARF
  mínimo.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar código 6015, prazo, e o valor mínimo de
  DARF vigente e como acumular abaixo dele.

### FISC-053 — Multa e juros por atraso
- **chunk_id:** `fiscal:apuracao:multa-atraso`
- **Regra (rascunho):** O pagamento em atraso do DARF gera **multa de mora**
  (a confirmar, tipicamente 0,33% por dia de atraso, limitada a 20%) e
  **juros pela taxa Selic acumulada** mais 1% no mês do pagamento. Os
  acréscimos são calculados sobre o valor do imposto devido.
- **Fonte oficial (a confirmar):** Lei nº 9.430/1996; regras de acréscimos
  legais da RFB.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar a multa de mora (0,33%/dia até 20%) e a
  regra de juros Selic + 1%.

### FISC-054 — Declaração anual (DIRPF)
- **chunk_id:** `fiscal:apuracao:declaracao-anual`
- **Regra (rascunho):** Independentemente do recolhimento mensal, as operações,
  posições e resultados precisam ser informados na **Declaração de Ajuste
  Anual (DIRPF)** (a confirmar), incluindo bens e direitos (posição em
  31/12), rendimentos isentos (ex.: dividendos, rendimentos de FII) e
  tributáveis. A apuração mensal não substitui a declaração anual.
- **Fonte oficial (a confirmar):** IN RFB da DIRPF do exercício.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar o que precisa ser declarado e a distinção
  entre recolhimento mensal e declaração anual.
