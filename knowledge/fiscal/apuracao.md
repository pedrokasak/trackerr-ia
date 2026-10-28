# Apuração, prejuízo e DARF — RASCUNHO (revisar · TRA-36)

> Rascunho gerado por IA. Confirmar vigência, números e fonte com
> profissional habilitado (CRC). Ver `README.md`.
>
> **Atualizado em 2026-10-28** (exercício 2027). Ver a linha *Conferência
> interna* de cada chunk.

### FISC-050 — Prejuízo compensável por modalidade
- **chunk_id:** `fiscal:apuracao:prejuizo-por-modalidade`
- **Regra (rascunho):** Prejuízos em renda variável podem ser compensados com
  **ganhos futuros da mesma modalidade**: prejuízo de operações comuns
  compensa ganhos de operações comuns (ações à vista e ETF); prejuízo de day
  trade compensa apenas ganhos de day trade; FII é tratado como categoria
  própria. A compensação não tem prazo de validade, mas precisa de controle
  contínuo mês a mês. Em mês de venda isenta (até R$ 20.000 em ações), o
  prejuízo antigo continua acumulando e o ganho isento não o consome.
- **Fonte oficial (a confirmar):** IN RFB nº 1.585/2015.
- **Conferência interna (2026-10-25):** de norma conhecida, **não relida na
  fonte oficial**.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar a separação por modalidade (comum vs.
  day trade vs. FII), que ETF e ações à vista dividem o mesmo saldo e que o
  prejuízo acumulado não expira.

### FISC-051 — Controle mensal e continuidade
- **chunk_id:** `fiscal:apuracao:controle-mensal`
- **Regra (rascunho):** A apuração de imposto sobre renda variável é
  **mensal**. O prejuízo de um mês é levado para abater ganhos dos meses
  seguintes. Quem não vende nada em um mês não tem imposto a pagar naquele mês,
  mas o controle do prejuízo acumulado deve continuar. O custo de cada ativo é
  o **preço médio ponderado** das compras, somando corretagem e emolumentos;
  vender não altera o preço médio da posição que sobra.
- **Fonte oficial (a confirmar):** IN RFB nº 1.585/2015.
- **Conferência interna (2026-10-25):** de norma conhecida, **não relida na
  fonte oficial**.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar a periodicidade mensal, o carregamento do
  prejuízo e o cálculo do custo médio.

### FISC-052 — DARF: código, prazo e valor mínimo
- **chunk_id:** `fiscal:apuracao:darf`
- **Regra (rascunho):** O imposto sobre ganho em renda variável é recolhido
  pelo próprio investidor via **DARF**, código **6015** (a confirmar), até o
  **último dia útil do mês seguinte** ao da apuração. Há um **valor mínimo**
  para emissão de DARF: abaixo de **R$ 10,00** o imposto não é emitido na hora
  e soma ao do mês seguinte. O ganho de capital em criptoativos usa outro
  código (4600, a confirmar).
- **Fonte oficial (a confirmar):** IN RFB nº 1.585/2015; regra geral de DARF
  mínimo.
- **Conferência interna (2026-10-25):** código 6015, prazo e mínimo de R$ 10
  são de norma conhecida, **não relidos na fonte oficial**. O 4600 para
  cripto também está **a confirmar**.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar os códigos 6015 e 4600, o prazo e o valor
  mínimo de DARF vigente e como acumular abaixo dele.

### FISC-053 — Multa e juros por atraso
- **chunk_id:** `fiscal:apuracao:multa-atraso`
- **Regra (rascunho):** O pagamento em atraso do DARF gera **multa de mora**
  (a confirmar, tipicamente 0,33% por dia de atraso, limitada a 20%) e
  **juros pela taxa Selic acumulada** mais 1% no mês do pagamento. Os
  acréscimos são calculados sobre o valor do imposto devido.
- **Fonte oficial (a confirmar):** Lei nº 9.430/1996; regras de acréscimos
  legais da RFB.
- **Conferência interna:** não verificado.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar a multa de mora (0,33%/dia até 20%) e a
  regra de juros Selic + 1%.

### FISC-054 — Declaração anual (DIRPF)
- **chunk_id:** `fiscal:apuracao:declaracao-anual`
- **Regra (rascunho):** Independentemente do recolhimento mensal, as operações,
  posições e resultados precisam ser informados na **Declaração de Ajuste
  Anual (DIRPF)**: a posição de bens e direitos em 31/12 (pelo custo de
  aquisição), os rendimentos isentos (ex.: dividendos dentro da isenção,
  rendimentos de FII) e os rendimentos de tributação exclusiva (ex.: JCP). A
  apuração mensal não substitui a declaração anual. O prazo e a
  obrigatoriedade do exercício 2027 ainda não foram publicados e **não devem
  ser afirmados**.
- **Fonte oficial (a confirmar):** IN RFB da DIRPF do exercício.
- **Conferência interna (2026-10-25):** a distribuição entre as fichas é de
  norma conhecida; prazo e piso de obrigatoriedade do exercício 2027 **não
  verificados**.
- **Status:** ⬜ pendente
- **Notas para o revisor:** confirmar o que precisa ser declarado e a distinção
  entre recolhimento mensal e declaração anual.

### FISC-055 — Imposto mínimo de altas rendas (IRPFM)
- **chunk_id:** `fiscal:apuracao:irpfm`
- **Regra (rascunho):** A partir do exercício 2027 (ano-calendário 2026) existe
  um **imposto mínimo** para quem recebeu **mais de R$ 600.000 no ano**,
  somando todos os rendimentos, inclusive os de tributação exclusiva, os
  isentos e os dividendos. A alíquota é de **10%** para rendimentos a partir de
  R$ 1.200.000; entre R$ 600.000 e R$ 1.200.000 ela cresce de forma linear,
  segundo a fórmula (rendimentos ÷ 60.000) − 10, em %. O imposto já pago na
  declaração e na fonte é deduzido. **Ganhos de bolsa entram na base**: o texto
  deduz os ganhos de capital, exceto os de operações em bolsa ou balcão
  organizado tributadas pelo ganho líquido. Lucros e dividendos aprovados até
  31/12/2025 ficam fora da base se pagos em 2026, 2027 e 2028 conforme o ato
  de aprovação.
- **Fonte oficial (a confirmar):** Lei nº 9.250/1995, art. 16-A (incluído pela
  Lei nº 15.270/2025).
- **Conferência interna (2026-10-25):** lido no texto da lei. O produto **não
  calcula** o IRPFM: se a renda do usuário for desse porte, avisa que ele
  precisa de cálculo completo.
- **Status:** ⬜ pendente
- **Notas para o revisor:** **prioridade**. Confirmar a fórmula, a base e como
  o imposto retido sobre dividendos (FISC-008) é deduzido.

### FISC-056 — Códigos de Bens e Direitos (ações e FII)
- **chunk_id:** `fiscal:apuracao:bens-direitos-codigos`
- **Regra (rascunho):** Na ficha "Bens e Direitos" da declaração, as **ações**
  (inclusive as listadas em bolsa) são informadas no **Grupo 03, Código 01**, e
  os **fundos de investimento imobiliário (FII)** no **Grupo 07, Código 03**.
  Para **ETF** e **criptoativos** não indicamos um código fixo: o usuário deve
  escolher na lista do programa da Receita. A tabela mudou na reorganização do
  exercício 2022; códigos antigos (como o "31" para ações) estão
  desatualizados.
- **Fonte oficial (a confirmar):** tabela de códigos de Bens e Direitos do
  programa do IRPF.
- **Conferência interna (2026-10-28):** os códigos de ação e FII vieram só de
  **matérias sobre a reorganização de 2022** (a Receita bloqueia o download do
  PDF da tabela), então estão como **fonte secundária**. Para o ETF as fontes
  se contradizem (apareceu como 07-06, 74 e 49), por isso ficou sem código.
- **Status:** ⬜ pendente
- **Notas para o revisor:** **prioridade**. Conferir os códigos de ação e FII
  na tabela do programa do exercício 2027 e dizer o código correto de ETF e
  de criptoativos.
