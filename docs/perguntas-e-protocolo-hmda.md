# Perguntas de pesquisa e protocolo do HMDA

O Entregável 2 descreve uma comparação supervisionada de decisões históricas de financiamento para compra de residência principal. Uma linha representa um pedido ou uma operação. Uma pessoa pode aparecer em mais de uma linha. O estudo mede generalização dentro do universo do HMDA 2024 e diferenças de previsão e erro entre grupos. A comparação é experimental e seus resultados servem para analisar decisões registradas.

## O que cada pergunta pode responder

| Pergunta do Entregável 2 | Como responder | Estado |
| --- | --- | --- |
| Acrescentar raça e etnia melhora a previsão ou muda as aprovações previstas por grupo? | Comparar A+B com A no mesmo teste e examinar ROC-AUC, PR-AUC da recusa, acurácia balanceada e taxas de aprovação e erro. | A rodada de 100 mil mostrou ganhos pequenos em ROC-AUC e mudanças maiores em erros de alguns grupos. Conferir a rodada integral. |
| A renda da região agrega informação além do perfil financeiro? | Comparar A+C com A nas mesmas partições. | Na rodada de 100 mil, a ROC-AUC mudou pouco e a PR-AUC da recusa caiu nos dois algoritmos. Conferir a rodada integral. |
| Perfis financeiros semelhantes recebem previsões diferentes ao mudar raça ou etnia? | Comparar a previsão do mesmo modelo nos mesmos perfis de teste, trocando um campo por vez e fixando as demais entradas. O CSV informa a diferença média na probabilidade e na aprovação prevista. | Implementado nos notebooks 02 e 03. Pendente nos relatórios oficiais. Mede a sensibilidade do modelo à troca de um rótulo. |
| A recusa histórica ou prevista para pedidos da categoria Black or African American aumenta em setores de renda superior? | Usar a dimensão raça × renda, comparar o grupo nas quatro faixas e conferir o número de pedidos, recusas observadas e recusas previstas em cada célula. | Implementado nos notebooks 02 e 03. Pendente nos relatórios oficiais. |
| A análise conjunta revela padrões que as análises separadas omitem? | Comparar A+B+C com A+B e A+C e examinar raça × renda e etnia × renda. | A comparação das configurações já existe na rodada de 100 mil. Os cruzamentos dependem da nova execução. |
| Os resultados são semelhantes entre algoritmos, instituições e períodos? | Comparar modelos na mesma partição e repetir o protocolo por instituição e por ano, com denominadores suficientes. A estabilidade temporal exige dados de outros anos e um teste temporal. | A comparação por algoritmo já existe. Instituições e períodos continuam em aberto. |

## Como ler as disparidades

Para raça, a comparação em destaque usa **Black or African American** e **White**. Para etnia, usa **Hispanic or Latino** e **Not Hispanic or Latino**. As duas dimensões podem se sobrepor. O HMDA não tem uma categoria equivalente a **parda** na classificação brasileira. Categorias publicadas como `Joint` ou `2 or more minority races` mantêm seus próprios rótulos.

Uma diferença de +13,66 pontos percentuais na recusa observada significa que a taxa do grupo foi 13,66 pontos maior do que a do grupo de referência naquele teste. As taxas descrevem os pedidos observados, com as variáveis disponíveis. Elas não fornecem uma porcentagem de racismo nem identificam o efeito causal sobre um solicitante. A simulação informa quanto um modelo já ajustado muda sua previsão para as mesmas entradas observadas, ao trocar apenas um rótulo. Ela não simula a decisão de um credor.

## Dados, campos e diferenças em relação ao plano inicial

- A primeira comparação preserva o conjunto A já utilizado na rodada de 100 mil. `loan_purpose` e `occupancy_type` atuam como filtros. `combined_loan_to_value_ratio` foi sugerido no Entregável 2, mas não integra A nesta comparação. `census_tract`, `state_code` e `county_code` ficam fora das entradas. A renda regional entra como percentual contínuo somente nas configurações com C.
- As faixas didáticas do novo relatório seguem o Entregável 2: abaixo de 50%, de 50% a menos de 80%, de 80% a menos de 120%, 120% ou mais e ausente. O relatório anterior agrupava as duas primeiras faixas. Os cortes mudam somente a apresentação por grupo. O valor contínuo continua sendo a entrada dos modelos com C.
- A comparação mantém o recorte validado de compra de residência principal e decisões 1, 2 ou 3. Exclusões adicionais por finalidade comercial, hipoteca reversa e linha aberta, a auditoria de duplicatas, os modelos simples de referência e a investigação formal de interações continuam como extensões previstas no Entregável 2.
- Os notebooks preservam o mesmo conjunto de teste para as oito combinações dentro de cada rodada. As duas escalas usam testes de tamanhos diferentes e podem compartilhar registros. A diferença entre os relatórios de 100 mil e integral é descritiva. Para isolar o efeito do volume de treino, seria necessário fixar um teste comum entre as rodadas.
- As referências de Ghoba e Colaner sobre pareamento e de Koeman et al. sobre crédito e contexto regional já constam em [referencias.bib](../referencias.bib). O pareamento descrito nos trabalhos relacionados é distinto da simulação de troca de rótulo usada neste projeto.

## Execução e entregas

O [notebook 02](../notebooks/02_treinamento_100k.ipynb) escreve em `reports/local/100k`. O [notebook 03](../notebooks/03_treinamento_recorte_integral.ipynb) escreve em `reports/local/full`. No Colab, os mesmos nomes ficam abaixo de `reports/colab`. Com os dois relatórios novos disponíveis, o segundo notebook executado gera `comparacao_100k_integral.csv`.

Cada rodada exporta métricas gerais, matriz de confusão em contagens, precisão, recall, F1, Brier score, calibração, taxas por grupo e diferenças frente a referências explícitas. Os gráficos incluem pizza da composição, barras de desempenho e disparidades, curva de calibração e barras de sensibilidade para perfis fixos. O `execucao.json` registra o tamanho efetivamente usado e as partições.

### Comparar a mesma solicitação em cenários diferentes

No modelo completo, cada perfil de teste é repetido em 12 cenários: três categorias de raça (branca, negra e indígena dos EUA/Alasca), duas categorias de etnia (latina e não latina) e dois níveis de renda do setor. Usamos 67% e 143% da renda familiar mediana da região de referência nas duas rodadas. São as medianas no ajuste da rodada de 100 mil para setores abaixo de 80% e a partir de 120%. Na comparação de renda, os dados do solicitante, os termos do pedido, o valor do imóvel e a instituição permanecem fixos.

O arquivo `perfis_pareados_cenarios.csv` mostra a probabilidade média de aprovação e as porcentagens previstas de aprovação e recusa. O `perfis_pareados_diferencas.csv` mostra mudanças em pontos percentuais e quantos perfis mudam de aprovação para recusa ou vice-versa. Os gráficos `perfis_pareados_*.png` apresentam raça, etnia, renda do setor e combinações em português. O gráfico combinado usa como referência uma pessoa branca, não latina, em setor de renda baixa. As trocas de rótulo e de renda são cenários do modelo. Algumas combinações podem ser raras na base e exigem cuidado na interpretação.

O notebook 03 grava em `reports/local/full` cinco contrastes selecionados para apresentação. Se os contrastes da rodada de 100 mil estiverem disponíveis, gera também um CSV e dois gráficos que colocam as rodadas lado a lado. As rodadas compartilham os cenários e os valores de renda, mas usam perfis de teste sorteados de suas respectivas partições. A comparação entre escalas é descritiva. Para o Overleaf, envie também `artigo_perfis_fixos_integral.tex` junto com as linhas LaTeX das duas tabelas integrais.

O artigo contém duas tabelas de desempenho e duas de conclusão. As duas tabelas de 100 mil usam os resultados já verificados. As duas do recorte integral passam a usar os números quando o notebook 03 gerar `artigo_metricas_linhas.tex` e `artigo_conclusao_linhas.tex`. Para compilar no Overleaf, envie esses dois arquivos para a mesma pasta do artigo.
