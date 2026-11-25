# Contexto do projeto

Projeto final de Ciência de Dados (CARCDSI) de João Gabriel de Faria Beserra e Lucas Alves de Melo Lins, do IFSP Campus Caraguatatuba. O manuscrito está em [artigo.tex](../artigo.tex), no formato SBC. A origem Git é [Luskahz/hmda-credit-fairness](https://github.com/Luskahz/hmda-credit-fairness), branch `master`.

## Pergunta de pesquisa

Como a inclusão de raça e etnia registradas e da renda relativa do setor censitário do imóvel, separadamente e em conjunto, afeta o desempenho preditivo e a distribuição dos erros entre grupos? O estudo usa decisões históricas de financiamento imobiliário nos EUA. Os resultados serão interpretados como associações no recorte estudado.

## Desenho experimental

- A unidade de análise é a solicitação. O recorte inicial é compra de residência principal no HMDA 2024. Em `action_taken`, os códigos 1 e 2 formam a classe de aprovação e o código 3 representa recusa. Os demais ficam fora do recorte.
- **A** inclui renda, idade e endividamento do solicitante, dados do financiamento e do imóvel, instituição e ano. **A+B** inclui raça e etnia. **A+C** inclui a renda familiar mediana do setor censitário em relação à região de referência. **A+B+C** inclui ambos os acréscimos.
- O resumo e a metodologia especificam regressão logística e floresta aleatória, totalizando oito combinações. A referência a “4 modelos diferentes” na introdução precisa ser alinhada com essas seções.
- Usar a mesma divisão estratificada, cerca de 80% para treino e 20% para teste. Tratar ausências, codificar atributos, ajustar modelos e escolher o limiar nas etapas de treino e validação. Excluir juros, custos finais e motivos de recusa para evitar vazamento do desfecho.
- Medir ROC-AUC, PR-AUC para recusas e acurácia balanceada. Por grupo demográfico e faixa de renda local, informar o número de pedidos, taxas de aprovação observada e prevista e taxas de falsos positivos e falsos negativos. Aplicar o mesmo limiar a todos os grupos em cada modelo.
- O HMDA público traz informação parcial sobre o histórico de crédito e os critérios das instituições. Interpretar estimativas de grupos pequenos com cautela.

## Estado e pendências

Os notebooks 02 e 03 agora simulam 12 combinações de raça, etnia e renda do setor em perfis fixos. Reexecutei a análise dos modelos completos com a amostra de 100 mil e confirmei 24 cenários e 62 contrastes nos dois algoritmos. Os novos CSVs e quatro gráficos em português estão em `reports/local/100k`. O [resumo dos cenários](resultados-treinamento.md#o-que-muda-nos-mesmos-20-mil-perfis) e o artigo incluem os resultados. A mesma análise no recorte integral ainda precisa ser executada.

Para a apresentação, o notebook 03 repete os cenários com os mesmos valores de renda relativa da rodada de 100 mil (67% e 143%). Ele exporta um resumo de cinco contrastes, os gráficos comparativos por algoritmo quando o CSV de 100 mil está disponível e o trecho `artigo_perfis_fixos_integral.tex` para o artigo. Ainda não há relatório da rodada integral em `reports/local/full`.

O material do Entregável 2 foi incorporado ao [protocolo de pesquisa](perguntas-e-protocolo-hmda.md), que registra as seis perguntas e o estado das análises. O notebook de treinamento combinado foi removido. As novas execuções usam [02 — 100 mil](../notebooks/02_treinamento_100k.ipynb) e [03 — recorte integral](../notebooks/03_treinamento_recorte_integral.ipynb). Eles geram relatórios separados, análise de disparidades, cruzamentos com renda e simulação de perfis fixos. As duas tabelas integrais do artigo serão preenchidas pelas saídas do notebook 03.

Os novos notebooks foram testados com um recorte de 12 mil linhas reais e florestas reduzidas. O teste confirmou os relatórios de cada pasta, a comparação entre as duas rodadas e os arquivos de linhas LaTeX. Os relatórios novos de 100 mil e do recorte integral ainda precisam ser executados nas escalas publicadas.

A primeira rodada local de treinamento terminou em 24/11/2026. Foram usadas 100 mil solicitações elegíveis e 20 mil registros de teste. A inclusão de atributos produziu ganhos globais modestos e mudanças mais pronunciadas em alguns erros por grupo. A [análise dos resultados](resultados-treinamento.md) reúne a tabela das oito combinações, as diferenças em relação a A e exemplos para o artigo e a apresentação.

O artigo e o roteiro de slides já incluem os resultados da primeira rodada e identificam a amostra de 100 mil e o teste de 20 mil. O notebook 03 usa as 3.794.116 solicitações elegíveis, com divisão estratificada de aproximadamente 60/20/20, e grava em `reports/local/full`. O notebook 02 grava em `reports/local/100k`. A bibliografia está em [referencias.bib](../referencias.bib) e reúne nove entradas, incluindo as três fontes do FFIEC citadas no manuscrito.

A tabela da primeira rodada foi conferida contra o CSV de métricas. O custo em tempo e memória da rodada integral ainda precisa ser medido neste PC. A compilação de conferência com o TeX Live local parou por falta da fonte Helvetica usada nas legendas do estilo SBC. O Overleaf é o ambiente escolhido para compilar o manuscrito.

Os relatórios antigos diretamente em `reports/local` registram a rodada de 100 mil. Para confirmar a rodada integral, executar todas as células do notebook 03 e conferir `sample_size: 3794116` em `reports/local/full/execucao.json`.

O pacote LaTeX da SBC está em [sbc-template.sty](../sbc-template.sty) e o estilo bibliográfico usado pelo artigo está em [sbc.bst](../sbc.bst).

A [proveniência dos dados](data-provenance.md) registra a edição escolhida e o arquivo bruto. Na próxima etapa, inspecionar tipos e códigos, aplicar os filtros de elegibilidade e registrar contagens e distribuição do alvo. Ainda faltam definir o tratamento das categorias demográficas e dos dados ausentes, os intervalos de renda relativa, a semente e a estratégia de validação.

## Continuidade entre chats

Em 24/11/2026, confirmamos que o notebook 01 concluiu o preparo local. O Parquet final contém 3.794.116 solicitações elegíveis, em acordo com os relatórios de validação. Com o Parquet pronto, os notebooks 02 e 03 podem ser executados diretamente no JupyterLab, em qualquer ordem.

A pedido do usuário, atualizar este arquivo a cada resposta sobre o projeto, inclusive quando a resposta for apenas uma orientação. Registrar as ações executadas, decisões, verificações e pendências pertinentes. Quando houver detalhes em outro documento de `docs/`, incluir aqui um resumo com o link.

O [guia didático de treinamento](guia-didatico-treinamento.md) explica o histórico, os recortes, o ajuste e a avaliação dos modelos. Também inclui um roteiro para a futura apresentação.

O [notebook 01](../notebooks/01_validacao_e_preparo.ipynb) processa o CSV em blocos e exporta o Parquet e as contagens de auditoria. Os notebooks 02 e 03 usam a semente 42 e escolhem um limiar por modelo na validação. Dentro de cada rodada, as oito combinações compartilham as partições. O [protocolo](perguntas-e-protocolo-hmda.md) registra as categorias e faixas de renda.

O ambiente local de treinamento fica em `.venv` e usa Python 3.12. No PowerShell, ativar com `.\.venv\Scripts\Activate.ps1`. No CMD, usar `.venv\Scripts\activate.bat`. Instalar as dependências com `python -m pip install -r requirements.txt` após ativar o ambiente. O [requirements.txt](../requirements.txt) registra as versões diretas verificadas no ambiente local.

Ao retomar o projeto em outro chat, ler este arquivo e a documentação vinculada. Atualizar o contexto quando o objetivo, o desenho ou o estado mudar. Registrar decisões de dados em `data-provenance.md` e usar arquivos temáticos em `docs/` para decisões que ganharem detalhes próprios. Datar decisões importantes e indicar os arquivos ou resultados que as sustentam.
