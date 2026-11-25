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

A conclusão do artigo descreve o que a análise permitirá avaliar. A tabela de resultados está em branco e os experimentos ainda precisam ser executados. A bibliografia está em [referencias.bib](../referencias.bib) e reúne nove entradas, incluindo as três fontes do FFIEC citadas no manuscrito.

O pacote LaTeX da SBC está em [sbc-template.sty](../sbc-template.sty) e o estilo bibliográfico usado pelo artigo está em [sbc.bst](../sbc.bst).

A [proveniência dos dados](data-provenance.md) registra a edição escolhida e o arquivo bruto. Na próxima etapa, inspecionar tipos e códigos, aplicar os filtros de elegibilidade e registrar contagens e distribuição do alvo. Ainda faltam definir o tratamento das categorias demográficas e dos dados ausentes, os intervalos de renda relativa, a semente e a estratégia de validação.

## Continuidade entre chats

Ao retomar o projeto em outro chat, ler este arquivo e a documentação vinculada. Atualizar o contexto quando o objetivo, o desenho ou o estado mudar. Registrar decisões de dados em `data-provenance.md` e usar arquivos temáticos em `docs/` para decisões que ganharem detalhes próprios. Datar decisões importantes e indicar os arquivos ou resultados que as sustentam.
