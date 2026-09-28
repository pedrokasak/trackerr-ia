"""
Corpus do guardrail de RI (TRA-239).

Frases escritas no estilo e com o vocabulario de releases, fatos relevantes
e relatorios de FII da B3 — nao copiadas de documentos reais, e sem nome de
empresa, para nao gravar no repositorio numero atribuido a alguem. O que
importa aqui e a FORMA: e nela que o guard do RAG pessoal errava ("venda da
subsidiaria", "vai investir", "recomendacao do conselho").

Fora do padrao `test_*.py` de proposito: e dado, nao teste.
"""

# Linguagem factual de documento de RI: tem que passar.
FACTUAL_RELEASE_PHRASES = [
    "A receita líquida consolidada atingiu R$ 12,3 bilhões no 2T26, alta de 8,1% em relação ao 2T25.",
    "O EBITDA ajustado somou R$ 4,2 bilhões, com margem de 34,1%, expansão de 1,2 p.p. na comparação anual.",
    "O lucro líquido recorrente foi de R$ 1,8 bilhão, 5,4% inferior ao mesmo período do ano anterior.",
    "A alavancagem, medida pela relação dívida líquida/EBITDA, encerrou o trimestre em 1,8x.",
    "O Conselho de Administração aprovou a distribuição de R$ 0,85 por ação em juros sobre capital próprio.",
    "A Companhia concluiu a venda de sua participação na subsidiária de distribuição de gás.",
    "Venda já concluída da participação na distribuidora.",
    "Os investimentos (capex) totalizaram R$ 2,1 bilhões, concentrados na expansão da capacidade produtiva.",
    "A administração reiterou o guidance de produção para 2026, entre 2,7 e 2,9 milhões de barris por dia.",
    "A Companhia vai investir R$ 2 bilhões na modernização de suas plantas até 2028.",
    "O índice de inadimplência acima de 90 dias foi de 2,9%, estável em relação ao trimestre anterior.",
    "A carteira de crédito expandida cresceu 9,7% em doze meses, para R$ 1,1 trilhão.",
    "O fluxo de caixa livre foi negativo em R$ 350 milhões, reflexo do maior consumo de capital de giro.",
    "O Conselho aprovou a recomendação de pagamento de dividendos intercalares.",
    "As vendas no conceito mesmas lojas cresceram 4,5% no trimestre.",
    "A Companhia recomprou 12,5 milhões de ações no âmbito do programa de recompra.",
    "A vacância física do portfólio encerrou o semestre em 6,3%.",
    "O fundo distribuiu R$ 1,05 por cota, equivalente a dividend yield anualizado de 11,2%.",
]

# Recomendacao dirigida ao leitor: nunca pode sair num resumo.
DIRECTED_RECOMMENDATIONS = [
    "Compre a ação antes do próximo balanço.",
    "Com esse resultado, você deveria vender suas ações.",
    "Recomendamos a compra do papel.",
    "Recomendo que você reduza a exposição ao setor.",
    "Vendam enquanto o preço está alto.",
    "Você pode aumentar sua posição aproveitando a queda.",
    "É hora de comprar: o resultado surpreendeu.",
    "Oportunidade de compra após o balanço.",
    "Invista agora antes da alta.",
]

# Preco-alvo proprio: nem o Trackerr nem o resumo podem estimar.
PRICE_TARGETS = [
    "Preço-alvo de R$ 45,00 para os próximos 12 meses.",
    "Potencial de valorização de 30% segundo a análise.",
    "Target price de US$ 18 para o ADR.",
]
