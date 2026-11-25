# Resultados da primeira rodada com o HMDA 2024

A rodada local usou 100.000 das 3.794.116 solicitações elegíveis, com semente 42. Foram 60.000 para ajuste, 20.000 para validação do limiar e 20.000 para teste. O teste continha 17.063 aprovações e 2.937 recusas. As oito combinações usaram as mesmas partições.

## Desempenho geral no teste

| Algoritmo | Atributos | ROC-AUC | PR-AUC para recusas | Acurácia balanceada |
| --- | --- | ---: | ---: | ---: |
| Regressão logística | A | 0,8645 | 0,6928 | 0,7875 |
| Regressão logística | A+B | 0,8690 | 0,6981 | **0,7928** |
| Regressão logística | A+C | 0,8649 | 0,6925 | 0,7878 |
| Regressão logística | A+B+C | 0,8697 | **0,6982** | 0,7908 |
| Floresta aleatória | A | 0,8679 | **0,6845** | 0,7958 |
| Floresta aleatória | A+B | 0,8700 | 0,6759 | 0,7981 |
| Floresta aleatória | A+C | 0,8665 | 0,6739 | 0,7965 |
| Floresta aleatória | A+B+C | **0,8704** | 0,6792 | **0,7997** |

Em negrito, os maiores valores por algoritmo em acurácia balanceada e PR-AUC, e a maior ROC-AUC geral.

- **Raça e etnia (A+B)** elevaram a ROC-AUC em 0,45 ponto percentual na regressão e 0,21 na floresta, em relação a A. A PR-AUC para recusas subiu 0,52 ponto na regressão e caiu 0,86 na floresta.
- **Renda relativa do setor (A+C)**, isoladamente, alterou a ROC-AUC em +0,04 ponto na regressão e −0,14 na floresta. A PR-AUC para recusas caiu 0,03 e 1,06 ponto, respectivamente.
- **A+B+C** apresentou a maior ROC-AUC nos dois algoritmos. Em relação a A, o ganho foi de 0,52 ponto na regressão e 0,25 na floresta. A maior acurácia balanceada foi a da floresta A+B+C. A maior PR-AUC para recusas foi a da regressão A+B+C.

## Mudanças nos erros por grupo

| Comparação com A | Grupo no teste | FPR em A → configuração | FNR em A → configuração |
| --- | --- | ---: | ---: |
| Regressão logística A+B+C | Black or African American no HMDA, 1.772 pedidos (467 recusados e 1.305 aprovados) | 28,27% → 20,34% | 16,32% → 28,66% |
| Floresta aleatória A+C | Renda relativa abaixo de 80%, 3.933 pedidos (679 recusados e 3.254 aprovados) | 35,64% → 30,19% | 12,75% → 18,35% |

Nos dois exemplos, caíram as aprovações previstas entre pedidos historicamente recusados e subiram as recusas previstas entre pedidos historicamente aprovados. Cada combinação usou seu próprio limiar, escolhido na validação e aplicado igualmente aos grupos. As mudanças nas taxas refletem a combinação entre a inclusão de atributos e o limiar selecionado para aquele modelo.

## Interpretação para o artigo e a apresentação

As maiores melhorias globais em relação à referência foram inferiores a um ponto percentual nas três métricas. Alguns grupos tiveram mudanças maiores na distribuição dos erros. Na regressão A+B+C, por exemplo, o grupo Black or African American teve redução de 7,92 pontos percentuais em FPR e aumento de 12,34 pontos em FNR. Esses são resultados descritivos de uma rodada, sem intervalos de confiança. O estudo usa decisões históricas como alvo e as mudanças nas taxas não estabelecem efeito causal.

Os dados para preencher a tabela do artigo e montar os slides estão em `reports/local/metricas_gerais.csv`, `reports/local/metricas_por_grupo.csv`, `reports/local/particoes.csv` e `reports/local/execucao.json`. Os gráficos da rodada também estão em `reports/local/`.

## O que muda nos mesmos 20 mil perfis

Os modelos A+B+C foram ajustados na amostra de 100 mil e avaliados nos mesmos 20 mil perfis de teste. Variamos os rótulos de raça e etnia e a renda relativa do setor, mantendo as demais entradas fixas. Setores de renda baixa e alta foram representados por 67% e 143% da renda familiar mediana da região de referência, as medianas das respectivas faixas na partição de ajuste.

| Comparação, com os demais atributos fixos | Aprovações previstas na regressão | Aprovações previstas na floresta |
| --- | ---: | ---: |
| Branca → negra, não latina, em setor de renda baixa | 80,71% → 66,27% | 83,27% → 77,24% |
| Branca → indígena dos EUA/Alasca, não latina, em setor de renda baixa | 80,71% → 78,83% | 83,27% → 81,57% |
| Não latina → latina, pessoa branca, em setor de renda baixa | 80,71% → 74,45% | 83,27% → 80,20% |
| Setor de renda baixa → alta, pessoa branca e não latina | 80,71% → 83,39% | 83,27% → 83,11% |
| Setor de renda baixa → alta, pessoa negra e latina | 55,75% → 61,14% | 75,04% → 75,57% |

Nos cenários negro e latino, a aprovação prevista subiu ao trocar para um setor de renda alta nos dois modelos. Ainda ficou abaixo do cenário branco, não latino e de renda baixa. O tamanho da mudança variou conforme os rótulos e o algoritmo. Os números são proporções de perfis classificados como aprovados após aplicar o limiar de cada modelo. A probabilidade média é uma medida diferente e consta no CSV.

Os [cenários](../reports/local/100k/perfis_pareados_cenarios.csv) trazem as taxas de aprovação e recusa. As [diferenças](../reports/local/100k/perfis_pareados_diferencas.csv) mostram os dois sentidos de mudança individual da classificação. Os quatro gráficos ficam em `reports/local/100k`, com os nomes `perfis_pareados_raca.png`, `perfis_pareados_etnia.png`, `perfis_pareados_renda.png` e `perfis_pareados_combinado.png`.
