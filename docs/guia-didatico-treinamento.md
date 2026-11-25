# Como preparamos os dados e treinamos os modelos com o HMDA 2024

Este guia acompanha o projeto de João Gabriel de Faria Beserra e Lucas Alves de Melo Lins, do IFSP Campus Caraguatatuba. Ele explica o caminho do CSV original até as tabelas e os gráficos que poderão ser usados no artigo e em uma apresentação da disciplina.

As novas execuções usam os notebooks 02 e 03, com 100 mil pedidos e recorte integral, respectivamente. O [protocolo atualizado](perguntas-e-protocolo-hmda.md) explica as perguntas do Entregável 2 e como usar os gráficos de diferenças de recusa, as pizzas de composição e as simulações de perfis fixos na apresentação.

Para apresentar os cenários de perfis fixos nas duas escalas, use `comparacao_perfis_regressao.png` e `comparacao_perfis_floresta.png` gerados pelo notebook 03. As barras mostram mudanças nas aprovações previstas em pontos percentuais para cinco contrastes. Diga quais atributos mudaram e qual cenário foi usado como referência.

## A pergunta que orienta o experimento

Queremos medir quanto **raça e etnia registradas** e **renda relativa do setor censitário** acrescentam à previsão de uma decisão histórica de financiamento imobiliário. Também queremos saber se a inclusão dessas informações muda os tipos de erro cometidos em grupos diferentes. Um modelo pode melhorar na média e, ao mesmo tempo, passar a errar mais para um grupo específico.

A unidade de análise é uma solicitação de financiamento. O modelo recebe atributos do pedido e estima a probabilidade de aprovação. O alvo vem da decisão registrada pela instituição no HMDA. A análise descreve associações no recorte estudado e depende das informações disponíveis na base pública. [Artigo do projeto](../artigo.tex)

## O que já foi preparado

1. Organizamos o manuscrito e os arquivos de bibliografia e formatação SBC no repositório `hmda-credit-fairness`.
2. Baixamos e verificamos o snapshot nacional do HMDA de 2024, congelado em 19/05/2025. O CSV extraído tem 99 colunas e cerca de 4,6 GB. A fonte, a versão e o SHA-256 do ZIP original estão registrados na documentação. [Proveniência dos dados](data-provenance.md)
3. Criamos o ambiente local `.venv` com Python 3.12.16. O pandas lê e transforma os dados, o PyArrow grava e lê o Parquet, o scikit-learn ajusta os modelos e o Matplotlib gera os gráficos. O JupyterLab permite executar os notebooks no PC.
4. Criamos dois notebooks para rodar no PC ou no Colab. O fluxo foi testado localmente com 12 mil linhas reais, usando uma configuração reduzida. O teste verificou o preparo, as oito combinações, a exportação das métricas e a geração dos gráficos. A execução local com o CSV completo foi interrompida durante o preparo para que o usuário pudesse rodar o fluxo no próprio terminal.

## Notebook 01 — da base bruta ao recorte

O notebook lê o CSV em blocos de 100 mil linhas e seleciona somente as colunas necessárias. O uso de blocos permite percorrer o arquivo completo sem colocar os 4,6 GB na memória de uma vez.

| Etapa | Regra |
| --- | --- |
| Ano | Manter registros com `activity_year = 2024`. |
| Finalidade | Manter `loan_purpose = 1`, compra de imóvel. |
| Ocupação | Manter `occupancy_type = 1`, residência principal. |
| Aprovação | `action_taken = 1` para financiamento contratado e `2` para aprovação sem contratação. Ambos recebem `approved = 1`. |
| Recusa | `action_taken = 3` recebe `approved = 0`. Os outros códigos ficam fora do recorte. |

Idade e razão dívida/renda são tratadas como categorias porque o LAR público inclui faixas nesses campos. As colunas numéricas são convertidas para tipos adequados e a renda relativa do setor é marcada como ausente quando não é positiva. Usamos as categorias agregadas de raça e etnia publicadas pelo HMDA, que podem refletir dados do solicitante e do co-solicitante.

O notebook valida se o recorte tem as duas classes, confere se o número de linhas gravadas bate com as contagens e cria `data/processed/hmda_2024_purchase.parquet`. Também exporta CSVs com contagens por etapa, ações registradas, distribuição do alvo, ausências e composição dos grupos demográficos. Essas tabelas respondem quantos registros entraram no estudo e como o recorte ficou composto. [Notebook 01](../notebooks/01_validacao_e_preparo.ipynb)

## Notebooks 02 e 03 — como os modelos aprendem

Na primeira rodada, sorteamos **100 mil solicitações elegíveis** com a semente **42**. O sorteio percorreu todo o Parquet. Para a próxima rodada, o notebook está configurado com `SAMPLE_SIZE = None` para incluir as **3.794.116 solicitações elegíveis**. Nos dois modos, as divisões estratificadas destinam aproximadamente 60% ao **ajuste**, 20% à **validação** e 20% ao **teste**. Todos os modelos usam as mesmas solicitações em cada partição. A execução completa grava os relatórios em `reports/local/full` no PC e `reports/colab/full` no Drive.

| Configuração | Dados usados para prever |
| --- | --- |
| **A** | Renda do solicitante, idade, razão dívida/renda, valor e prazo do empréstimo, valor do imóvel, tipo de empréstimo, posição da garantia, método de construção, número de unidades, instituição e ano. |
| **A+B** | A mais raça e etnia agregadas. |
| **A+C** | A mais o percentual da renda familiar mediana do setor censitário em relação à região de referência. |
| **A+B+C** | A com os dois acréscimos. |

Para comparar o efeito das variáveis, partimos de A. Por exemplo, **A+B menos A** mede a mudança na métrica ao acrescentar raça e etnia para o mesmo algoritmo. Campos ligados ao desfecho, como juros, custos finais e motivos de recusa, ficam fora das entradas para evitar vazamento de informação.

1. **Pré-processamento.** Calculamos a mediana das variáveis numéricas na partição de ajuste e usamos esse valor para preencher ausências. Na regressão logística, também padronizamos essas variáveis. Nas categóricas, representamos ausências, aplicamos one-hot encoding e agrupamos categorias raras segundo a frequência no ajuste. A transformação aprende essas regras uma vez e as reaplica na validação e no teste.
2. **Ajuste.** A regressão logística combina os atributos para estimar a probabilidade de aprovação. A floresta aleatória combina previsões de várias árvores de decisão. Rodamos ambos os algoritmos para cada configuração, somando oito ajustes.
3. **Validação.** Cada ajuste produz uma probabilidade. Escolhemos na validação o limiar de aprovação que maximiza a acurácia balanceada. Quanto menor o limiar, mais pedidos tendem a ser classificados como aprovados. Cada combinação tem seu próprio limiar, aplicado igualmente a todos os grupos.
4. **Teste.** Usamos os modelos já ajustados e os limiares já escolhidos para gerar as métricas finais na partição reservada. [Rodada de 100 mil](../notebooks/02_treinamento_100k.ipynb) e [rodada integral](../notebooks/03_treinamento_recorte_integral.ipynb)

## Como interpretar os resultados

| Métrica | Pergunta que ela responde |
| --- | --- |
| ROC-AUC | Com que qualidade o modelo ordena aprovações acima de recusas ao variar o limiar? |
| PR-AUC para recusas | Qual é o desempenho na relação entre precisão e revocação quando o objetivo é identificar recusas? Nesta métrica, recusa é a classe positiva. |
| Acurácia balanceada | Qual é a média entre a proporção de aprovações corretamente classificadas e a proporção de recusas corretamente classificadas? |
| FPR por grupo | Entre os pedidos recusados na decisão histórica, quantos foram previstos como aprovados? |
| FNR por grupo | Entre os pedidos aprovados na decisão histórica, quantos foram previstos como recusados? |

A avaliação por grupo mostra também o tamanho de cada grupo e as taxas de aprovação observada e prevista. A raça, a etnia e a renda relativa são avaliadas separadamente. As faixas propostas para a renda do setor são **abaixo de 80%**, **de 80% até menos de 120%** e **120% ou mais**. Compare FPR e FNR com seus denominadores, especialmente quando há poucas recusas ou aprovações no grupo. Quando o denominador de uma taxa é zero, o valor fica em branco no CSV.

## Como executar e localizar os arquivos

No CMD, abra o JupyterLab pelo ambiente do projeto.

```cmd
cd /d "%USERPROFILE%\Documents\artigo"
.venv\Scripts\activate.bat
python -m jupyterlab
```

Se o Parquet ainda não existir, execute o notebook 01 até aparecer **Parquet pronto**. Execute o notebook 02 para a rodada de 100 mil e o 03 para a rodada integral. Os dois podem rodar em qualquer ordem. No Colab, ajuste o caminho do CSV na célula de configuração caso ele esteja em outra pasta do Drive. No PC, os relatórios ficam em `reports/local/100k` e `reports/local/full`. No Colab, as mesmas subpastas ficam em `reports/colab` no Drive.

| Saída | Uso no artigo ou na apresentação |
| --- | --- |
| `validacao_etapas.csv` e `validacao_alvo.csv` | Mostrar o fluxo dos filtros e a distribuição de aprovações e recusas. |
| `validacao_ausencias.csv` | Explicar o tratamento das informações ausentes. |
| `particoes.csv` e `execucao.json` | Registrar tamanho da amostra, divisão, semente e parâmetros da rodada. |
| `metricas_gerais.csv` e `metricas_gerais.png` | Comparar as oito combinações e preencher a tabela do artigo. |
| `metricas_por_grupo.csv` e `erros_*.png` | Examinar diferenças de FPR e FNR com as contagens de cada grupo. |

## Roteiro para a futura apresentação

| Slide | O que explicar | Material para usar |
| --- | --- | --- |
| 1. Pergunta | O que muda quando acrescentamos raça, etnia e renda do setor às previsões? | Título, autores e pergunta de pesquisa. |
| 2. Dados e recorte | De onde vêm os registros e como a decisão vira o alvo. | Fonte oficial e contagens por etapa do notebook 01. |
| 3. Desenho da comparação | O que A, A+B, A+C e A+B+C acrescentam. | Tabela das configurações. |
| 4. Treinamento e validação | Como a mesma amostra é dividida e como o limiar é escolhido. | Diagrama com ajuste, validação e teste e contagens de cada partição. |
| 5. Desempenho geral | Quanto as métricas mudaram em relação a A dentro de cada algoritmo. | Gráfico geral e diferenças calculadas a partir do CSV. |
| 6. Erros por grupo | Onde FPR e FNR mudaram e quantos pedidos sustentam cada taxa. | Um gráfico com os grupos mais informativos e seus denominadores. |
| 7. Conclusão | Resposta à pergunta de pesquisa, com o recorte e as limitações da base. | Resultados da rodada escolhida para o artigo. |

### Texto para os slides da primeira rodada

- **Dados e partições.** 3.794.116 pedidos elegíveis no HMDA 2024. Nesta rodada, foram sorteados 100.000. Usamos 60.000 para ajuste, 20.000 para validação e 20.000 para teste.
- **Desempenho geral.** No teste de 20 mil pedidos, a floresta A+B+C alcançou ROC-AUC de 0,8704 e acurácia balanceada de 79,97%. A regressão A+B+C teve a maior PR-AUC para recusas, de 0,6982. Os melhores ganhos sobre A ficaram abaixo de um ponto percentual.
- **Erros por grupo.** Na regressão A+B+C, entre os 1.772 pedidos da categoria *Black or African American*, a FPR caiu de 28,27% para 20,34% e a FNR subiu de 16,32% para 28,66%, em relação a A.
- **Conclusão e próxima rodada.** A primeira rodada mostrou mudanças maiores nos erros de alguns grupos do que nas métricas gerais. A próxima rodada incluirá todo o recorte elegível, com aproximadamente 60% para ajuste, 20% para validação e 20% para teste. [Tabela e análise da rodada de 100 mil](resultados-treinamento.md)
