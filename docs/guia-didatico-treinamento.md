# Como preparamos os dados e treinamos os modelos com o HMDA 2024

Este guia acompanha o projeto de João Gabriel de Faria Beserra e Lucas Alves de Melo Lins, do IFSP Campus Caraguatatuba. Ele explica o caminho do CSV original até as tabelas e os gráficos que poderão ser usados no artigo e em uma apresentação da disciplina.

O treinamento usa todos os pedidos elegíveis com os 18 campos válidos. O [protocolo](perguntas-e-protocolo-hmda.md) explica as perguntas do Entregável 2 e os gráficos para a apresentação.

Para apresentar os cenários de perfis fixos, use os gráficos `perfis_pareados_*.png` e o CSV `perfis_pareados_resumo_apresentacao.csv`. Explique quais atributos mudaram e qual cenário foi usado como referência.

## A pergunta que orienta o experimento

Queremos medir quanto **raça e etnia registradas** e **renda relativa do setor censitário** acrescentam à previsão de uma decisão histórica de financiamento imobiliário. Também queremos saber se a inclusão dessas informações muda os tipos de erro cometidos em grupos diferentes. Um modelo pode melhorar na média e, ao mesmo tempo, passar a errar mais para um grupo específico.

A unidade de análise é uma solicitação de financiamento. O modelo recebe atributos do pedido e estima a probabilidade de aprovação. O alvo vem da decisão registrada pela instituição no HMDA. A análise descreve associações no recorte estudado e depende das informações disponíveis na base pública. [Artigo do projeto](../artigo.tex)

## O que já foi preparado

1. Organizamos o manuscrito e os arquivos de bibliografia e formatação SBC no repositório `hmda-credit-fairness`.
2. Baixamos e verificamos o snapshot nacional do HMDA de 2024, congelado em 19/05/2025. O CSV extraído tem 99 colunas e cerca de 4,6 GB. A fonte, a versão e o SHA-256 do ZIP original estão registrados na documentação. [Proveniência dos dados](data-provenance.md)
3. Criamos o ambiente local `.venv` com Python 3.12.16. O pandas lê e transforma os dados, o PyArrow grava e lê o Parquet, o scikit-learn ajusta os modelos e o Matplotlib gera os gráficos. O JupyterLab permite executar os notebooks no PC.
4. O notebook 01 prepara a base e o notebook 02 treina as oito combinações. O fluxo integral foi executado localmente e gerou as métricas e os gráficos.

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

O notebook aplica a auditoria dos 18 campos aos pedidos elegíveis, confere se as duas classes estão presentes e grava `data/processed/valid_18/hmda_2024_purchase_valid18.parquet`. Na mesma pasta, exporta as contagens por etapa, os campos inválidos ou indisponíveis e a composição do recorte validado. No Colab, o notebook 01 também precisa de `scripts/contar_linhas_completas.py` na pasta do projeto. [Notebook 01](../notebooks/01_validacao_e_preparo.ipynb)

## Notebook 02 — como os modelos aprendem

O notebook usa as **2.910.642 solicitações elegíveis com os 18 campos válidos** e a semente **42**. A divisão estratificada reserva aproximadamente 70% para ajuste, 20% para validação e 10% para teste. Os modelos compartilham essas partições. Os relatórios ficam em `reports/local/full_valid18_702010` no PC e `reports/colab/full_valid18_702010` no Drive.

| Configuração | Dados usados para prever |
| --- | --- |
| **A** | Renda do solicitante, idade, razão dívida/renda, valor e prazo do empréstimo, valor do imóvel, tipo de empréstimo, posição da garantia, método de construção, número de unidades, instituição e ano. |
| **A+B** | A mais raça e etnia agregadas. |
| **A+C** | A mais o percentual da renda familiar mediana do setor censitário em relação à região de referência. |
| **A+B+C** | A com os dois acréscimos. |

Para comparar o efeito das variáveis, partimos de A. Por exemplo, **A+B menos A** mede a mudança na métrica ao acrescentar raça e etnia para o mesmo algoritmo. Campos ligados ao desfecho, como juros, custos finais e motivos de recusa, ficam fora das entradas para evitar vazamento de informação.

1. **Pré-processamento.** A regressão logística padroniza as variáveis numéricas. As categóricas são codificadas e as categorias raras são agrupadas pela frequência no ajuste. As transformações são aprendidas no ajuste e reaplicadas na validação e no teste. Os campos já chegam válidos do preparo, então o notebook dispensa imputação.
2. **Ajuste.** A regressão logística combina os atributos para estimar a probabilidade de aprovação. A floresta aleatória combina previsões de várias árvores de decisão. Rodamos ambos os algoritmos para cada configuração, somando oito ajustes.
3. **Validação.** Cada ajuste produz uma probabilidade. Escolhemos na validação o limiar de aprovação que maximiza a acurácia balanceada. Quanto menor o limiar, mais pedidos tendem a ser classificados como aprovados. Cada combinação tem seu próprio limiar, aplicado igualmente a todos os grupos.
4. **Teste.** Usamos os modelos já ajustados e os limiares já escolhidos para gerar as métricas finais na partição reservada. [Notebook 02](../notebooks/02_treinamento_recorte_integral.ipynb)

## Como interpretar os resultados

| Métrica | Pergunta que ela responde |
| --- | --- |
| ROC-AUC | Com que qualidade o modelo ordena aprovações acima de recusas ao variar o limiar? |
| PR-AUC para recusas | Qual é o desempenho na relação entre precisão e revocação quando o objetivo é identificar recusas? Nesta métrica, recusa é a classe positiva. |
| Acurácia balanceada | Qual é a média entre a proporção de aprovações corretamente classificadas e a proporção de recusas corretamente classificadas? |
| FPR por grupo | Entre os pedidos recusados na decisão histórica, quantos foram previstos como aprovados? |
| FNR por grupo | Entre os pedidos aprovados na decisão histórica, quantos foram previstos como recusados? |

A avaliação por grupo mostra também o tamanho de cada grupo e as taxas de aprovação observada e prevista. A raça, a etnia e a renda relativa são avaliadas separadamente. As faixas propostas para a renda do setor são **abaixo de 50%**, **de 50% até menos de 80%**, **de 80% até menos de 120%** e **120% ou mais**. Compare FPR e FNR com seus denominadores, especialmente quando há poucas recusas ou aprovações no grupo. Quando o denominador de uma taxa é zero, o valor fica em branco no CSV.

## Como executar e localizar os arquivos

No CMD, abra o JupyterLab pelo ambiente do projeto.

```cmd
cd /d "%USERPROFILE%\Documents\artigo"
.venv\Scripts\activate.bat
python -m jupyterlab
```

Se o Parquet ainda não existir, execute o notebook 01 até aparecer **Parquet pronto**. Depois, execute o notebook 02. No Colab, ajuste o caminho do CSV se ele estiver em outra pasta do Drive.

| Saída | Uso no artigo ou na apresentação |
| --- | --- |
| `validacao_etapas.csv` e `validacao_alvo.csv` | Mostrar o fluxo dos filtros e a distribuição de aprovações e recusas. |
| `validacao_invalidos_por_campo.csv` | Mostrar os campos inválidos ou indisponíveis que levaram à exclusão de pedidos. |
| `particoes.csv` e `execucao.json` | Registrar tamanho do recorte, divisão, semente e parâmetros. |
| `metricas_gerais.csv` e `metricas_gerais.png` | Comparar as oito combinações e preencher a tabela do artigo. |
| `metricas_por_grupo.csv` e `erros_*.png` | Examinar diferenças de FPR e FNR com as contagens de cada grupo. |

## Roteiro para a futura apresentação

| Slide | O que explicar | Material para usar |
| --- | --- | --- |
| 1. Pergunta | O que muda quando acrescentamos raça, etnia e renda do setor às previsões? | Título, autores e pergunta de pesquisa. |
| 2. Dados e recorte | De onde vêm os registros e como a decisão vira o alvo. | Fonte oficial e contagens por etapa do notebook 01. |
| 3. Desenho da comparação | O que A, A+B, A+C e A+B+C acrescentam. | Tabela das configurações. |
| 4. Treinamento e validação | Como o recorte é dividido e como o limiar é escolhido. | Diagrama com ajuste, validação e teste e contagens de cada partição. |
| 5. Desempenho geral | Quanto as métricas mudaram em relação a A dentro de cada algoritmo. | Gráfico geral e diferenças calculadas a partir do CSV. |
| 6. Erros por grupo | Onde FPR e FNR mudaram e quantos pedidos sustentam cada taxa. | Um gráfico com os grupos mais informativos e seus denominadores. |
| 7. Conclusão | Resposta à pergunta de pesquisa, com o recorte e as limitações da base. | Resultados do recorte integral. |

### Resultados anteriores para os slides

Os números abaixo descrevem o recorte anterior, que mantinha pedidos elegíveis com campos indisponíveis. Substitua-os pelos resultados de `reports/local/full_valid18_702010` após treinar com o recorte validado.

- **Dados e partições.** O recorte de 3.794.116 pedidos foi dividido em 2.276.469 para ajuste, 758.823 para validação e 758.824 para teste.
- **Desempenho geral.** A regressão logística A+B+C liderou as três métricas no teste, com ROC-AUC de 0,8786, PR-AUC para recusas de 0,7129 e acurácia balanceada de 80,32%.
- **Erros por grupo.** A diferença de recusas previstas entre negros e brancos foi de 24,00 pontos percentuais na regressão e 19,49 na floresta, na configuração A+B+C.
- **Perfis fixos.** No cenário negro e latino, trocar o setor de renda baixa pelo de renda alta elevou as aprovações previstas em 5,42 pontos na regressão e 5,67 na floresta. [Resultados](resultados-treinamento.md)
