# Resultados do recorte integral do HMDA 2024

Os resultados abaixo se referem ao recorte anterior, que incluía pedidos elegíveis com campos indisponíveis. O fluxo atual usa exclusivamente pedidos válidos nos 18 campos. As métricas desse novo recorte ainda dependem da execução completa do notebook 02.

## Recorte integral

A rodada integral usou 3.794.116 solicitações. Foram 2.276.469 para ajuste, 758.823 para validação e 758.824 para teste. A regressão logística A+B+C liderou as três métricas no teste integral, com ROC-AUC de 0,8786, PR-AUC de 0,7129 e acurácia balanceada de 80,32%. Na floresta, A+B liderou ROC-AUC (0,8660) e acurácia balanceada (79,89%), e A+C liderou PR-AUC para recusas (0,6655).

Na simulação integral de 20 mil perfis fixos, a troca de branca para negra em setor de renda baixa reduziu a aprovação prevista em 14,58 pontos percentuais na regressão e 12,01 na floresta. As diferenças para o cenário indígena foram de −5,96 e −5,80 pontos, e para a mudança de etnia não latina para latina, de −5,45 e −6,24. No cenário negro e latino, mover para setor de renda alta elevou as aprovações previstas em 5,42 e 5,67 pontos, respectivamente. O resultado ainda ficou abaixo da referência branca, não latina e de renda baixa.

Os resultados anteriores foram exportados em `reports/local/full`. O fluxo atual grava em `reports/local/full_valid18_702010`. O [artigo](../artigo.tex) traz duas tabelas com os resultados anteriores.
