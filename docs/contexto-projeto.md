# Contexto do projeto

Projeto final de Ciência de Dados (CARCDSI) de João Gabriel de Faria Beserra e Lucas Alves de Melo Lins, do IFSP Campus Caraguatatuba. O manuscrito está em [artigo.tex](../artigo.tex), no formato SBC. A origem Git é [Luskahz/hmda-credit-fairness](https://github.com/Luskahz/hmda-credit-fairness), branch `master`.

## Pergunta de pesquisa

Como a inclusão de raça e etnia registradas e da renda relativa do setor censitário do imóvel, separadamente e em conjunto, afeta o desempenho preditivo e a distribuição dos erros entre grupos? O estudo usa decisões históricas de financiamento imobiliário nos EUA. Os resultados serão interpretados como associações no recorte estudado.

## Desenho experimental

- A unidade de análise é a solicitação. O recorte inicial é compra de residência principal no HMDA 2024. Em `action_taken`, os códigos 1 e 2 formam a classe de aprovação e o código 3 representa recusa. Os demais ficam fora do recorte.
- **A** inclui renda, idade e endividamento do solicitante, dados do financiamento e do imóvel, instituição e ano. **A+B** inclui raça e etnia. **A+C** inclui a renda familiar mediana do setor censitário em relação à região de referência. **A+B+C** inclui ambos os acréscimos.
- São quatro configurações de atributos em dois algoritmos, totalizando oito combinações.
- Usar a mesma divisão estratificada, aproximadamente 70% para ajuste, 20% para validação e 10% para teste. Tratar ausências, codificar atributos, ajustar modelos e escolher o limiar nas etapas de treino e validação. Excluir juros, custos finais e motivos de recusa para evitar vazamento do desfecho.
- Medir ROC-AUC, PR-AUC para recusas e acurácia balanceada. Por grupo demográfico e faixa de renda local, informar o número de pedidos, taxas de aprovação observada e prevista e taxas de falsos positivos e falsos negativos. Aplicar o mesmo limiar a todos os grupos em cada modelo.
- O HMDA público traz informação parcial sobre o histórico de crédito e os critérios das instituições. Interpretar estimativas de grupos pequenos com cautela.

## Estado e pendências em 25/11/2026

O projeto mantém um único fluxo, com todos os pedidos elegíveis que passaram na validação dos 18 campos. O snapshot HMDA 2024 tem 12.229.298 linhas e 99 colunas. Os filtros do estudo produziram 3.794.116 pedidos elegíveis. A auditoria encontrou 2.910.642 pedidos com os 18 campos válidos (76,71%) e descartou 883.474 com pelo menos um valor inválido ou indisponível. O recorte validado contém 2.570.870 aprovações e 339.772 recusas. [Proveniência e auditoria](data-provenance.md)

O [notebook 01](../notebooks/01_validacao_e_preparo.ipynb) aplica o [script de validação](../scripts/contar_linhas_completas.py) e gera `data/processed/valid_18/hmda_2024_purchase_valid18.parquet`. Em 25/11/2026, restauramos o ZIP do HMDA e o CSV bruto no caminho esperado, com o SHA-256 conferido. Execute o notebook 01 para recriar o Parquet e os CSVs de auditoria. No Colab, o script de validação também precisa estar na pasta `scripts/` do projeto no Drive.

O [notebook 02](../notebooks/02_treinamento_recorte_integral.ipynb) lê apenas o Parquet validado, sem imputação. O arquivo `reports/local/full_valid18_702010/particoes.csv` confirma 2.037.448 pedidos para ajuste, 582.129 para validação e 291.065 para teste. O notebook ajusta regressão logística e floresta aleatória nas quatro configurações, escolhe o limiar na validação e usa 12 cenários de perfis fixos. Os dois valores de renda relativa usados nos cenários são 67% e 143%.

Na conferência, a pasta `reports/local/full_valid18_702010` continha apenas `particoes.csv`. A conclusão das oito combinações ainda precisa ser confirmada com `execucao.json`, `metricas_gerais.csv` e os demais relatórios. O [artigo](../artigo.tex) e o [resumo dos resultados](resultados-treinamento.md) ainda exibem as métricas da execução anterior com 3.794.116 pedidos elegíveis. Incorporar ao artigo e ao pitch os resultados do recorte validado quando a execução estiver completa.

O manuscrito usa [sbc-template.sty](../sbc-template.sty), [sbc.bst](../sbc.bst) e [referencias.bib](../referencias.bib). A compilação de conferência no TeX Live local parou por falta da fonte Helvetica do estilo. O Overleaf é o ambiente escolhido para compilar o artigo.

## Material para o pitch

- [O que eu entendi](oque-eu-entendi.md) consolida a escolha do tema, o dataset, os conjuntos A/B/C e o preparo. O Passo 5 usa dois pedidos fictícios e a ideia de nota mínima para explicar a escolha do ponto de corte. A primeira árvore ilustra a floresta. Os resultados ficam para o próximo passo.
- A [árvore completa do HMDA em PDF](arvore-de-decisao-hmda.pdf) tem 85 páginas, 2.681 nós e 1.341 folhas. Começa no nó 0 e tem links nos ramos Sim e Não. Também está [em Markdown](arvore-de-decisao-hmda.md). É a primeira árvore de A+B+C reproduzida com a divisão anterior, que tinha 1.746.384 pedidos de ajuste. Pode ser usada para explicar como ler uma árvore.
- O exemplo do Titanic ensina a ler os ramos de uma árvore. Os arquivos locais desse exemplo não estavam presentes na conferência. [Recurso da disciplina no ECAD](https://ecad.ifspcaraguatatuba.edu.br/mod/resource/view.php?id=267991)
- O [guia didático](guia-didatico-treinamento.md) traz o roteiro para os slides. O [protocolo](perguntas-e-protocolo-hmda.md) relaciona cada pergunta do estudo às análises correspondentes.

## Continuidade entre chats

O Passo 6 das [notas do pitch](oque-eu-entendi.md) responde às seis perguntas com três gráficos em `docs/figuras-conclusao/`. O gráfico de recusas observadas usa os 291.065 pedidos da partição de teste atual e mostra os dois grupos de raça por faixa de renda. Os gráficos de desempenho e perfis fixos reproduzem números da execução anterior documentados no artigo. Os resultados previstos da divisão atual ainda dependem da conclusão do notebook.

O [protocolo de pesquisa](perguntas-e-protocolo-hmda.md) está em texto corrido. As perguntas, formas de comparação e métricas anteriores foram preservadas, com a indicação de qual execução gerou os resultados.

O Passo 5 das notas do pitch foi ampliado em linguagem simples. Usa uma ficha fictícia para explicar pesos e categorias na regressão, um caminho real da árvore para explicar a floresta e versões da mesma ficha para distinguir as oito combinações de treinamento dos 12 cenários de perfis fixos.

As notas do pitch explicam a divisão estratificada com 100 pedidos fictícios, dos quais 90 aprovados e 10 recusados. O exemplo mostra como as pilhas de treino, validação e teste mantêm a proporção e explica que a semente 42 permite repetir o sorteio.

A pedido do usuário, atualizar este arquivo a cada resposta sobre o projeto, inclusive orientações. Registrar ações, decisões, verificações e pendências pertinentes. Vincular documentos de `docs/` que tragam mais detalhes.

O ambiente local é `.venv` com Python 3.12. No PowerShell, use `.\.venv\Scripts\Activate.ps1`. No CMD, use `.venv\Scripts\activate.bat`. As dependências estão em [requirements.txt](../requirements.txt). Para abrir os notebooks localmente, execute `python -m jupyterlab` com o ambiente ativo.
