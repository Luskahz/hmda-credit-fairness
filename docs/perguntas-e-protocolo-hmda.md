# Perguntas de pesquisa e protocolo do HMDA

O Entregável 2 descreve uma comparação supervisionada de decisões históricas de financiamento para compra de residência principal. Uma linha representa um pedido ou uma operação. Uma pessoa pode aparecer em mais de uma linha. O estudo mede generalização dentro do universo do HMDA 2024 e diferenças de previsão e erro entre grupos. A comparação é experimental e seus resultados servem para analisar decisões registradas.

## O que cada pergunta pode responder

Primeiro, queremos saber se acrescentar raça e etnia melhora a previsão geral ou muda quais pedidos aparecem como aprovados em cada grupo. Para isso, comparamos A+B com A nos mesmos pedidos de teste. Na execução anterior, a ROC-AUC subiu de 0,8747 para 0,8777 na regressão logística e de 0,8597 para 0,8660 na floresta aleatória. Para verificar se a renda da região acrescenta informação ao perfil financeiro, comparamos A+C com A. Na mesma execução, a ROC-AUC passou de 0,8747 para 0,8750 na regressão e de 0,8597 para 0,8617 na floresta.

Também queremos saber se o mesmo perfil financeiro recebe uma previsão diferente quando mudamos raça ou etnia. Nesse caso, usamos o mesmo modelo e o mesmo pedido. Trocamos um campo por vez e mantemos as outras informações iguais. O relatório mostra quanto mudou, em média, a probabilidade de aprovação e quantos pedidos foram classificados como aprovados em cada cenário. O artigo já traz essa análise para a execução anterior.

Outra pergunta é se a recusa observada ou prevista para os pedidos do grupo *Black or African American* aumenta em setores de renda superior. Para investigar isso, separamos os pedidos desse grupo pelas quatro faixas de renda do setor e verificamos quantos pedidos, recusas observadas e recusas previstas aparecem em cada faixa. O cruzamento raça e renda está em `metricas_por_grupo.csv`. Para ver se aparecem padrões quando juntamos as informações, comparamos A+B+C com A+B e A+C e examinamos também os cruzamentos de raça e etnia com renda.

Por fim, podemos comparar regressão e floresta usando a mesma partição de teste. A comparação por instituição e por período está em aberto. Para comparar instituições, precisamos de registros suficientes em cada uma. Para saber se os resultados se mantêm com o passar do tempo, precisamos de dados de outros anos e de um teste que reserve um período posterior ao treino.

## Como ler as disparidades

Para raça, a comparação em destaque usa **Black or African American** e **White**. Para etnia, usa **Hispanic or Latino** e **Not Hispanic or Latino**. As duas dimensões podem se sobrepor. O HMDA não tem uma categoria equivalente a **parda** na classificação brasileira. Categorias publicadas como `Joint` ou `2 or more minority races` mantêm seus próprios rótulos.

Uma diferença de +12,95 pontos percentuais na recusa observada significa que a taxa do grupo foi 12,95 pontos maior do que a do grupo de referência naquele teste. As taxas descrevem os pedidos observados, com as variáveis disponíveis. Elas não fornecem uma porcentagem de racismo nem identificam o efeito causal sobre um solicitante. A simulação informa quanto um modelo já ajustado muda sua previsão para as mesmas entradas observadas, ao trocar apenas um rótulo. Ela não simula a decisão de um credor.

## Dados, campos e diferenças em relação ao plano inicial

Usamos `loan_purpose` e `occupancy_type` para selecionar pedidos de compra de residência principal. Mantemos as decisões 1, 2 ou 3. O campo `combined_loan_to_value_ratio` foi sugerido no Entregável 2 e ficou fora de A. Os campos `census_tract`, `state_code` e `county_code` também ficaram fora das entradas.

Nos modelos com C, a renda relativa do setor entra como percentual contínuo. Nos relatórios por grupo, usamos faixas para facilitar a leitura. Elas são abaixo de 50%, de 50% a menos de 80%, de 80% a menos de 120% e 120% ou mais. O relatório pode separar valores ausentes quando houver. As faixas servem para apresentar as comparações entre grupos. Os modelos continuam recebendo o percentual original.

As oito combinações usam os mesmos pedidos de ajuste, validação e teste. O Entregável 2 também prevê extensões como exclusões adicionais por finalidade comercial, hipoteca reversa e linha aberta, auditoria de duplicatas, modelos simples de referência e investigação formal de interações.

As referências de Ghoba e Colaner sobre pareamento e de Koeman et al. sobre crédito e contexto regional estão em [referencias.bib](../referencias.bib). Neste projeto, a simulação usa cópias da mesma ficha e troca um campo por vez. O pareamento dos trabalhos relacionados usa observações distintas com características semelhantes.

## Execução e entregas

O fluxo atual seleciona pedidos elegíveis com os 18 campos válidos, conforme a auditoria compartilhada com o notebook 01. Os resultados numéricos citados acima descrevem a execução anterior com os 3.794.116 pedidos elegíveis.

O [notebook 02](../notebooks/02_treinamento_recorte_integral.ipynb) divide os pedidos validados em 70% para treinamento, 20% para validação e 10% para teste, com a semente 42. Os relatórios ficam em `reports/local/full_valid18_702010` no PC e em `reports/colab/full_valid18_702010` no Colab.

Cada rodada exporta métricas gerais, matriz de confusão em contagens, precisão, recall, F1, Brier score, calibração, taxas por grupo e diferenças frente a referências explícitas. Os gráficos incluem pizza da composição, barras de desempenho e disparidades, curva de calibração e barras de sensibilidade para perfis fixos. O `execucao.json` registra o tamanho efetivamente usado e as partições.

### Comparar a mesma solicitação em cenários diferentes

No modelo completo, cada perfil sorteado do teste é repetido em 12 cenários. Combinamos três categorias de raça, duas de etnia e dois níveis de renda relativa do setor. Os níveis de renda são fixados em 67% e 143% da mediana da região de referência. Os dados financeiros, os termos do pedido, o valor do imóvel e a instituição permanecem fixos.

O arquivo `perfis_pareados_cenarios.csv` mostra a probabilidade média de aprovação e as porcentagens previstas de aprovação e recusa. O `perfis_pareados_diferencas.csv` mostra mudanças em pontos percentuais e quantos perfis mudam de aprovação para recusa ou vice-versa. Os gráficos `perfis_pareados_*.png` apresentam raça, etnia, renda do setor e combinações em português. O gráfico combinado usa como referência uma pessoa branca, não latina, em setor de renda baixa. As trocas de rótulo e de renda são cenários do modelo. Algumas combinações podem ser raras na base e exigem cuidado na interpretação.

O notebook 02 grava cinco contrastes selecionados para a apresentação em `perfis_pareados_resumo_apresentacao.csv`.

O [artigo](../artigo.tex) incorpora os resultados do recorte com os 18 campos válidos e divisão 70/20/10. A conclusão reúne as seis perguntas, a comparação usada para responder a cada uma e o que foi observado. Os números da execução anterior citados no início deste protocolo servem como registro histórico.
