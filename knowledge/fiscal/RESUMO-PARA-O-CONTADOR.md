# Resumo do Imposto de Renda sobre Investimentos — para revisão do contador

> **Este documento é só pra você revisar.** Ele não entra no aplicativo, não
> é publicado e não é lido por ninguém além de quem está cuidando da parte
> fiscal do projeto. Se preferir, pode editar direto neste arquivo, ou marcar
> em vermelho o que precisar de correção e devolver.

## O que é o Trakker

O Trakker é um aplicativo que ajuda a pessoa a acompanhar os investimentos
dela (ações, fundos imobiliários, renda fixa, etc.) num só lugar. Queremos
que ele também explique, de forma simples, **como funciona o imposto de
renda** de cada tipo de investimento — não para calcular o imposto de
ninguém automaticamente ainda, só para **explicar a regra geral**, do jeito
que um material educativo explicaria.

Por exemplo: se a pessoa tem uma ação e vende com lucro, o aplicativo poderá
mostrar um texto tipo *"lucro na venda de ações costuma ser tributado em
15%, e há isenção se o total vendido no mês for até R$ 20.000"* — sem
calcular o imposto dela específico, só explicando a regra.

## Por que estamos te mandando isso

Preparamos um primeiro rascunho dessas explicações, mas **foi escrito com
ajuda de um sistema automático de texto**, não por um profissional de
contabilidade. Isso quer dizer que:

- Os números (alíquotas, valores de isenção, prazos) podem estar
  **desatualizados ou errados**. O sistema que escreveu não sabe com certeza
  o que está em vigor hoje.
- Nada disso vai aparecer pra nenhum usuário do aplicativo até você (ou
  outro contador) confirmar que está certo.
- Se alguma regra mudou recentemente e o texto abaixo não reflete isso, é
  exatamente esse tipo de erro que precisamos que você aponte.

Sua parte é simples: ler cada item, confirmar se o número e a regra estão
certos hoje (ou corrigir), e apontar a lei/norma certa se a que colocamos
estiver errada ou incompleta. Não precisa reescrever o texto todo — só
corrigir o que estiver errado.

---

## 1. Ações

**Venda comum (não é day trade):**
Lucro na venda de ações é tributado em **15%**. Mas há uma isenção: se a
pessoa vender até **R$ 20.000 em ações no mês** (somando tudo que ela vendeu
naquele mês, não só o lucro), não paga imposto nenhum sobre esse lucro. Essa
isenção **não vale para day trade**.

**Day trade** (compra e venda da mesma ação no mesmo dia):
Tributado em **20%**, sem direito a nenhuma isenção.

**Imposto retido na hora da venda ("dedo-duro"):**
A corretora já desconta uma parte pequena na hora da venda, só pra avisar a
Receita que a operação aconteceu — **0,005%** do valor vendido em operação
comum, **1%** do lucro em day trade. Esse valor descontado entra como
abatimento no imposto que a pessoa vai pagar depois.

**Como e quando pagar:**
O imposto é apurado **mês a mês** e pago pela própria pessoa através de uma
guia (DARF), até o último dia útil do mês seguinte.

**Dividendos:**
Hoje são **isentos** de imposto de renda para quem recebe. (Atenção: esse é
um dos pontos mais discutidos por propostas de reforma tributária — se isso
mudou, é importante avisar.)

**Juros sobre capital próprio (JCP):**
Diferente do dividendo, o JCP **é tributado**, em **15%**, já descontado na
hora do pagamento.

---

## 2. Fundos Imobiliários (FIIs)

**Rendimento mensal que o fundo distribui:**
Pode ser **isento** de imposto, mas só se três condições forem cumpridas ao
mesmo tempo: o fundo ser negociado em bolsa, ter no mínimo 50 cotistas, e a
pessoa ter menos de 10% das cotas do fundo. Se alguma dessas condições não
for cumprida, o rendimento deixa de ser isento.

**Venda das cotas com lucro:**
Diferente do rendimento mensal, **vender a cota com lucro é sempre
tributado**, em **20%**. Não existe a isenção de R$ 20.000/mês que existe
para ações — isso é uma diferença que costuma confundir os usuários, então é
importante que o texto deixe isso bem claro.

**Prejuízo:**
Se a pessoa perde dinheiro vendendo cotas de um FII, esse prejuízo só pode
ser usado para abater lucro de **outro FII no futuro** — não abate lucro de
ações, por exemplo.

**Como e quando pagar:** mesma lógica das ações — apuração mensal, DARF até
o último dia útil do mês seguinte.

---

## 3. ETFs (fundos negociados em bolsa)

**ETF de ações:**
Lucro na venda é tributado em **15%**, mas **sem** a isenção de R$ 20.000
que existe para ações individuais — outro ponto que costuma gerar confusão
e vale reforçar no texto.

**ETF de renda fixa:**
Tem uma lógica diferente: a alíquota varia de acordo com o prazo médio dos
títulos que o fundo carrega, não segue a mesma tabela usada em ações.
Precisamos que você confirme como funciona essa variação hoje.

---

## 4. Fundos de investimento (fundos abertos, come-cotas)

**"Come-cotas":**
Em muitos fundos de renda fixa e multimercado, duas vezes por ano (maio e
novembro) o governo já desconta uma parte do imposto automaticamente,
reduzindo a quantidade de cotas da pessoa. A alíquota usada nesse desconto
antecipado costuma ser **15%** para fundos de longo prazo e **20%** para
fundos de curto prazo.

**Alíquota final, na hora de resgatar:**
Depende de há quanto tempo o dinheiro está aplicado — quanto mais tempo,
menor a alíquota (de 22,5% até 180 dias até 15% acima de 720 dias). O que já
foi descontado no come-cotas é abatido desse valor final.

**Atenção especial — fundos fechados, exclusivos e investimento no
exterior:** essa parte teve mudança de lei recente (2023) e é a que mais
precisa da sua confirmação. Preferimos não publicar nada sobre esse tema até
você validar, porque o risco de informação errada aqui é maior.

---

## 5. Renda fixa (Tesouro Direto, CDB, LCI/LCA, debêntures)

**Regra geral (Tesouro, CDB e parecidos):**
Segue uma tabela que **diminui o imposto quanto mais tempo o dinheiro fica
aplicado**: 22,5% até 180 dias, 20% de 181 a 360 dias, 17,5% de 361 a 720
dias, e 15% acima de 720 dias. O banco/corretora já desconta na hora do
resgate.

**LCI e LCA:**
Hoje são **isentas** de imposto de renda para pessoa física.

**Debêntures incentivadas** (as de projeto de infraestrutura, específicas —
diferentes de uma debênture comum):
Também **isentas**. Debêntures comuns seguem a tabela normal, igual ao
Tesouro/CDB.

**Resgate muito rápido (menos de 30 dias):**
Além do imposto de renda, pode incidir um imposto extra e diferente (IOF)
que vai diminuindo conforme os dias passam, até zerar no 30º dia.

---

## 6. Regras gerais de apuração (valem para todos os investimentos de renda variável — ações, FII, ETF)

- **O prejuízo não se perde**: se a pessoa perde dinheiro num mês, pode usar
  esse prejuízo pra abater lucro dos meses seguintes, sem prazo de validade
  — mas só dentro da mesma categoria (prejuízo de ação abate lucro de ação;
  prejuízo de day trade só abate lucro de day trade).
- **É mês a mês**: todo mês em que há lucro, calcula-se o imposto daquele
  mês. Se não vendeu nada, não há imposto naquele mês.
- **A guia de pagamento (DARF)** tem um valor mínimo — abaixo disso, o valor
  fica acumulado até o mês em que atingir o mínimo.
- **Atraso no pagamento** gera multa e juros sobre o valor do imposto.
- **Declaração anual**: independente do que foi pago mês a mês, todo ano a
  pessoa precisa declarar a posição de bens (o que tem em 31 de dezembro) e
  os rendimentos recebidos (incluindo os isentos, como dividendos) na
  declaração de ajuste anual.

---

## O que precisamos que você confirme

Pra cada item acima, precisamos saber:

1. **O número está certo hoje?** (alíquota, valor de isenção, prazo)
2. **Alguma lei mudou isso recentemente** e o texto ainda não reflete?
3. **A lei ou norma que citamos como fonte está certa?** (veja a lista
   abaixo — algumas nós só arriscamos um "candidato" de lei, sem ter
   certeza)

Não precisa formatar nada nem devolver em nenhum modelo específico — pode
escrever direto num e-mail, ou anotar em cima deste arquivo mesmo.

---

## Fontes que usamos (a confirmar com você)

Essas são as leis e normas que apontamos como base de cada regra acima. Não
temos certeza de que estão certas ou completas — é exatamente isso que
precisamos que você valide:

- Lei nº 9.249/1995 — isenção de dividendos (art. 10) e tributação de JCP
  (art. 9º)
- Lei nº 9.430/1996 — regras de multa e juros por atraso
- Lei nº 11.033/2004 — regra geral de tributação de renda variável (ações,
  FII, ETF), incluindo a isenção de R$ 20.000/mês em ações (art. 3º)
- Lei nº 11.196/2005 — condições da isenção de rendimento de FII
- Lei nº 11.311/2006 — isenção de LCI/LCA
- Lei nº 12.431/2011 — isenção de debêntures incentivadas
- Lei nº 13.043/2014 — tributação de ETF de renda fixa
- Lei nº 14.754/2023 — mudanças recentes em fundos fechados, exclusivos e
  investimento no exterior (a parte que mais precisa da sua atenção)
- Decreto nº 6.306/2007 — regulamento do IOF (resgates de curtíssimo prazo)
- Instrução Normativa RFB nº 1.585/2015 — regras operacionais: código de
  DARF, prazos, apuração mensal, retenção na fonte

Se alguma dessas leis já foi revogada ou substituída por outra mais recente,
é essa a correção mais importante que você pode nos dar.

---

**Alguma dúvida sobre o próprio aplicativo** (não sobre a parte fiscal) pode
ser encaminhada pra quem está desenvolvendo — este arquivo é só a parte que
precisa do seu olhar como contador.
