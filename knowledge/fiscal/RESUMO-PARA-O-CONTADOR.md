# Resumo do Imposto de Renda sobre Investimentos — para revisão do contador

> **Este documento é só pra você revisar.** Ele não entra no aplicativo, não
> é publicado e não é lido por ninguém além de quem está cuidando da parte
> fiscal do projeto. Se preferir, pode editar direto neste arquivo, ou marcar
> em vermelho o que precisar de correção e devolver.

> **Versão 0.2, atualizada em 28/10/2026 para o exercício 2027 (ano-calendário
> 2026).** A primeira versão (agosto) foi escrita sem acesso às fontes. Nesta,
> o que já foi lido na lei ou na Receita está dito como **"lido na fonte"**; o
> que é de norma conhecida mas **ainda não foi relido** e o que veio **só de
> imprensa ou corretora** também está identificado. Os itens que mais
> precisam do seu olhar estão marcados como **PRIORIDADE**.

## O que é o Trackerr

O Trackerr é um aplicativo que ajuda a pessoa a acompanhar os investimentos
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

*Lido na fonte (Lei 11.033/2004, planalto.gov.br):* as alíquotas de 15% e
20% e a isenção de R$ 20.000 ("para o conjunto de ações"). A MP 1.303/2025, que
previa alíquota única de 17,5% ou 18%, **perdeu a validade em 08/10/2025**;
17,5% e 18% **não** são regra vigente.

**Imposto retido na hora da venda ("dedo-duro"):**
A corretora já desconta uma parte pequena na hora da venda, só pra avisar a
Receita que a operação aconteceu — **0,005%** do valor vendido em operação
comum (*lido na fonte*), **1%** do lucro em day trade (*só fonte secundária*;
a lei remete a outro texto). Esse valor descontado entra como abatimento no
imposto que a pessoa vai pagar depois.

**Como e quando pagar:**
O imposto é apurado **mês a mês** e pago pela própria pessoa através de uma
guia (DARF, código 6015), até o último dia útil do mês seguinte. DARF abaixo de
R$ 10,00 não é emitido na hora: soma ao mês seguinte. *(Código, prazo e mínimo
são de norma conhecida, ainda não relidos na fonte.)*

**Dividendos — PRIORIDADE (a regra mudou em 2026):**
Os dividendos continuam, em regra, isentos. Mas, **desde janeiro de 2026**
(Lei 15.270/2025, que incluiu o art. 6º-A na Lei 9.250/1995), quando uma
mesma empresa paga a uma mesma pessoa física **mais de R$ 50.000 no mês**,
há retenção de **10% sobre o total pago** (não só sobre o que passa de
R$ 50.000). Vários pagamentos da mesma empresa no mês se somam. Ficam fora
os lucros apurados até 2025 com distribuição aprovada até 31/12/2025.
*(Lido na fonte.)* Além disso, os dividendos entram na base do imposto mínimo
de altas rendas (seção 7).

**Juros sobre capital próprio (JCP):**
Diferente do dividendo, o JCP **é tributado**, em **15%**, já descontado na
hora do pagamento, em tributação **exclusiva**. Na declaração vai em
"Rendimentos Sujeitos à Tributação Exclusiva/Definitiva", e não em
"Rendimentos Isentos". O valor que a corretora credita costuma ser o
**líquido**; o bruto e o imposto retido estão no informe de rendimentos.
*(15% é de norma conhecida, ainda não relido na fonte.)*

---

## 2. Fundos Imobiliários (FIIs)

**Rendimento mensal que o fundo distribui — PRIORIDADE:**
Pode ser **isento** de imposto, mas só se as condições forem cumpridas ao
mesmo tempo: as cotas serem negociadas em bolsa ou balcão organizado (*lido na
fonte*, Lei 11.033, art. 3º, III), a pessoa ter menos de 10% das cotas do
fundo e o fundo ter um mínimo de cotistas. **Aqui precisamos de você:** a
versão de agosto dizia 50 cotistas, mas a Lei 15.270/2025, ao definir o
imposto mínimo, repete a isenção com exigência de **ao menos 100 cotistas**.
Não afirmamos nenhum dos dois números até você confirmar qual vale para 2026.

**Venda das cotas com lucro:**
Diferente do rendimento mensal, **vender a cota com lucro é sempre
tributado**, em **20%** (*lido na fonte*: Lei 8.668/1993, art. 18; Fiagro,
art. 20-D). Não existe a isenção de R$ 20.000/mês que existe para ações — isso
é uma diferença que costuma confundir os usuários, então é importante que o
texto deixe isso bem claro. A versão de agosto citava a Lei 11.033 como fonte
dessa alíquota; a fonte correta é a Lei 8.668.

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

Que o ETF caia na regra geral de bolsa (15%) é entendimento da Receita, de
norma conhecida e **ainda não relido na fonte**. O prejuízo de ETF compensa
ganho de operações comuns, o mesmo saldo das ações à vista (a confirmar).

**ETF de renda fixa:**
Tem uma lógica diferente: a alíquota varia de acordo com o prazo médio dos
títulos que o fundo carrega, não segue a mesma tabela usada em ações.
Precisamos que você confirme como funciona essa variação hoje.

**Ticker terminado em "11" — ETF, FII ou unit?**
O final "11" não diz sozinho o que o ativo é. BOVA11 e IVVB11 são ETFs; SANB11
é uma **unit** (um pacote de ações negociado junto); outros são FIIs. O
aplicativo trata a unit como ação (com a isenção de R$ 20.000), o ETF como
ETF (15%, sem a isenção) e o FII como FII (20%). Classificar errado muda o
imposto estimado. **Precisamos que você confirme o tratamento da unit** para a
isenção e para a compensação de prejuízo.

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
- **A guia de pagamento (DARF)** tem um valor mínimo (R$ 10,00) — abaixo
  disso, o valor fica acumulado até o mês em que atingir o mínimo.
- **Atraso no pagamento** gera multa e juros sobre o valor do imposto.
- **Declaração anual**: independente do que foi pago mês a mês, todo ano a
  pessoa precisa declarar a posição de bens (o que tem em 31 de dezembro, pelo
  custo) e os rendimentos recebidos: os isentos em "Rendimentos Isentos" e os
  de tributação exclusiva, como o JCP, na ficha própria. O prazo e a
  obrigatoriedade do exercício 2027 ainda não foram publicados, por isso o
  texto não afirma datas.

---

## 7. Imposto mínimo de altas rendas (novo, exercício 2027) — PRIORIDADE

A Lei 15.270/2025 criou, a partir do exercício 2027, um **imposto mínimo**
para quem recebeu **mais de R$ 600.000 no ano**, somando todos os
rendimentos, inclusive os de tributação exclusiva, os isentos e os
dividendos. A alíquota é de 10% a partir de R$ 1.200.000; entre R$ 600.000 e
R$ 1.200.000 cresce linearmente: alíquota em % = (rendimentos ÷ 60.000) − 10.
O imposto já pago é deduzido. **Ganhos de bolsa entram na base** (o texto
deduz os ganhos de capital, exceto os de operações em bolsa tributadas pelo
ganho líquido). *(Lido na fonte: Lei 9.250/1995, art. 16-A.)* O aplicativo
**não calcula** esse imposto; apenas avisa que quem tem renda desse porte
precisa de cálculo completo. Precisamos que você confirme a leitura da base.

## 8. Códigos de Bens e Direitos — PRIORIDADE

Na ficha "Bens e Direitos", o aplicativo orienta: **ações no Grupo 03, Código
01** e **FII no Grupo 07, Código 03**. Para ETF e criptoativos **não indicamos
código**, porque as fontes se contradizem (o ETF apareceu como 07-06, 74 e 49).
Esses códigos vieram só de **matérias sobre a reorganização da tabela em
2022**, porque a Receita bloqueia o download do PDF da tabela; o aplicativo
manda o usuário conferir na lista do programa. A versão de agosto usava um
código antigo para ações ("31"). **Precisamos que você confirme os códigos na
tabela do exercício 2027 e diga o de ETF e o de criptoativos.**

## 9. Criptoativos (novo)

Ganho tributado a **15%**; vendas de até **R$ 35.000 no mês** podem ficar
isentas (Lei 9.250/1995, art. 22, II, *lido na fonte*; que valha para cripto
é entendimento da Receita, *fonte secundária*). A Receita não prevê
compensação de prejuízo. DARF com código próprio (4600, a confirmar). Confirme
a aplicação do limite, as faixas de alíquota e o código.

## 10. Previdência privada (PGBL e VGBL) (novo)

O aplicativo **explica e não recomenda**: não diz que vale a pena aportar,
nem indica valor, plano ou instituição. O PGBL permite deduzir os aportes da
base de cálculo, **até 12% dos rendimentos computados na base**, só na
declaração completa (*lido na fonte*: Lei 9.532/1997, art. 11); o imposto
incide sobre o valor total no resgate. O VGBL não é dedutível. Confirme as
regras e **avise se algum trecho soar como recomendação**.

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
- Lei nº 9.250/1995 — art. 6º-A (retenção de 10% sobre dividendos acima de
  R$ 50 mil por mês) e art. 16-A (imposto mínimo de altas rendas), ambos
  incluídos pela Lei nº 15.270/2025; art. 22 (alienação de bens e direitos
  até R$ 35 mil no mês)
- Lei nº 9.430/1996 — regras de multa e juros por atraso
- Lei nº 9.532/1997, art. 11 — limite de dedução do PGBL; Lei nº
  11.053/2004 — regimes de tributação do resgate (candidata, a confirmar)
- Lei nº 8.668/1993, arts. 18 e 20-D — alíquota de FII e Fiagro
- Lei nº 11.033/2004 — regra geral de tributação de renda variável (ações,
  ETF), incluindo as alíquotas (art. 2º), a isenção de R$ 20.000/mês em ações
  (art. 3º) e a tabela regressiva de renda fixa (art. 1º)
- Medida Provisória nº 1.303/2025 — **perdeu a validade em 08/10/2025**; citada
  só para registrar que 17,5% e 18% não são regra vigente
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
