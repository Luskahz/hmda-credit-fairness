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

O preparo histórico de 2024 gerou `data/processed/valid_18/hmda_2024_purchase_valid18.parquet` com o [script de validação](../scripts/contar_linhas_completas.py). O notebook 01 foi removido em 26/11/2026. Use os notebooks 03 e 04 para as novas rodadas.

O [notebook 02](../notebooks/02_treinamento_recorte_integral.ipynb) lê apenas o Parquet validado, sem imputação. O arquivo `reports/local/full_valid18_702010/particoes.csv` confirma 2.037.448 pedidos para ajuste, 582.129 para validação e 291.065 para teste. O notebook ajusta regressão logística e floresta aleatória nas quatro configurações, escolhe o limiar na validação e usa 12 cenários de perfis fixos. Os dois valores de renda relativa usados nos cenários são 67% e 143%.

A execução 70/20/10 terminou e gerou o manifesto e as métricas das oito combinações. As conclusões, tabelas e gráficos do [artigo](../artigo.tex) usam o teste de 291.065 pedidos. Os números da execução anterior permanecem como registro histórico nos documentos que a identificam.

O manuscrito usa [sbc-template.sty](../sbc-template.sty), [sbc.bst](../sbc.bst) e [referencias.bib](../referencias.bib). A compilação de conferência no TeX Live local parou por falta da fonte Helvetica do estilo. O Overleaf é o ambiente escolhido para compilar o artigo.

## Material para o pitch

- [O que eu entendi](oque-eu-entendi.md) consolida a escolha do tema, o dataset, os conjuntos A/B/C e o preparo. O Passo 5 usa dois pedidos fictícios e a ideia de nota mínima para explicar a escolha do ponto de corte. A primeira árvore ilustra a floresta. Os resultados ficam para o próximo passo.
- A [árvore completa do HMDA em PDF](arvore-de-decisao-hmda.pdf) tem 85 páginas, 2.681 nós e 1.341 folhas. Começa no nó 0 e tem links nos ramos Sim e Não. Também está [em Markdown](arvore-de-decisao-hmda.md). É a primeira árvore de A+B+C reproduzida com a divisão anterior, que tinha 1.746.384 pedidos de ajuste. Pode ser usada para explicar como ler uma árvore.
- O exemplo do Titanic ensina a ler os ramos de uma árvore. Os arquivos locais desse exemplo não estavam presentes na conferência. [Recurso da disciplina no ECAD](https://ecad.ifspcaraguatatuba.edu.br/mod/resource/view.php?id=267991)
- O [guia didático](guia-didatico-treinamento.md) traz o roteiro para os slides. O [protocolo](perguntas-e-protocolo-hmda.md) relaciona cada pergunta do estudo às análises correspondentes.

## Continuidade entre chats

No fluxo multiano, o notebook 03 valida e salva cada ano de 2018 a 2025 em um Parquet separado. O notebook 04 lê esses oito arquivos e usa os pedidos de treino de todos os anos no mesmo ajuste, preservando o ano para a avaliação. O fluxo dispensa um CSV único com todos os registros.

Os ZIPs de 2018 a 2025 estão presentes. Ainda não há Parquets anuais nem relatórios multiano. O próximo passo é abrir o JupyterLab pelo `.venv`, executar todas as células do notebook 03 e, quando o preparo terminar, executar o notebook 04.

Os três gráficos do artigo foram redesenhados com perguntas no título, explicações curtas em cada painel, legenda por modelo, valores nas barras e unidades simples. O gráfico de recusas distingue a decisão registrada da previsão. O de perfis fixos explica os dois lados do zero. Os números continuam vindo dos relatórios de HMDA 2024 com divisão 70/20/10.

O desenho multiano passa a perguntar como o desempenho e as disparidades de recusa variaram de 2018 a 2025. Os nove ZIPs disponíveis cobrem 2017 a 2025, mas 2017 tem layout de 45 campos e não satisfaz o recorte comum de 18 colunas. Por isso, o fluxo comparável usa oito anos. O [notebook 03](../notebooks/03_preparo_hmda_por_ano.ipynb) valida cada ZIP e grava um Parquet por ano. O [notebook 04](../notebooks/04_treinamento_hmda_multiano.ipynb) combina as partições de treino, valida em conjunto e exporta resultados gerais, resultados anuais e variações de disparidade. O notebook 02 documenta a análise histórica de 2024 do artigo e depende do Parquet original.

O fluxo multiano foi testado em uma pasta temporária com 65 mil linhas brutas de cada snapshot e florestas reduzidas. O teste confirmou a leitura dos oito anos, as oito combinações, os CSVs e os quatro gráficos. O preparo e o treinamento com os arquivos completos ainda não foram executados.

No fluxo multiano, a regressão logística é incremental e a floresta divide o ajuste por blocos com dados de vários anos. Essa implementação limita o uso de memória e difere da floresta de 2024, que usava a base inteira como origem do sorteio de cada árvore. O teste anual usa os pedidos reservados de cada ano e um limiar conjunto por combinação. A tendência racial é medida como a diferença de recusa entre negros e brancos, em pontos percentuais, com as contagens dos grupos. Uma diminuição dessa diferença sustenta uma conclusão descritiva sobre a disparidade, sem identificar mudança na discriminação. [Desenho multiano](perguntas-e-protocolo-hmda.md#extensão-multiano)

O usuário selecionou 2018, 2019, 2020, 2021 e 2022 para a extensão temporal. Os snapshots oficiais de 2023, 2024 e 2025 também existem. Em 26/11/2026, conferimos os cabeçalhos de 2023 e 2025 por leitura parcial dos ZIPs e confirmamos 99 colunas, incluindo as 18 do estudo. O CSV de 2024 já está no projeto e é a base do artigo atual. Os ZIPs de 2023, 2024 e 2025 foram baixados e verificados por inteiro em 26/11/2026. Os tamanhos e hashes constam na [proveniência dos dados](data-provenance.md). [Portal oficial do HMDA](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/)

O acervo histórico do HMDA nos Arquivos Nacionais dos EUA começa em 1981. O formato por pedido ou operação usado para estudar decisões de crédito começou em 1990, com primeira divulgação em outubro de 1991. O snapshot mais antigo no portal moderno é 2017. Para o formato usado no projeto, o ponto de partida é 2018. Conferimos que todos os CSVs baixados de 2018 a 2022 têm cabeçalho, 99 colunas e os mesmos nomes das 18 colunas usadas em 2024. Essa conferência cobre a estrutura do CSV. A validade dos valores e a comparabilidade dos códigos ainda devem ser verificadas por ano. [Acervo histórico](https://www.archives.gov/research/electronic-records/housing#rg82) · [Guia de dados do CFPB](https://github.com/cfpb/HMDA_Data_Science_Kit/blob/master/readme.md)

Para extensão temporal, verificamos os ZIPs nacionais LAR dos snapshots oficiais de [2020](https://files.ffiec.cfpb.gov/static-data/snapshot/2020/2020_public_lar_csv.zip), [2021](https://files.ffiec.cfpb.gov/static-data/snapshot/2021/2021_public_lar_csv.zip) e [2022](https://files.ffiec.cfpb.gov/static-data/snapshot/2022/2022_public_lar_csv.zip). Os servidores retornaram HTTP 200 para os três arquivos, com cerca de 1,46 GB, 1,52 GB e 0,88 GB comprimidos. Em 26/11/2026, baixamos e verificamos os ZIPs nacionais de 2017 a 2022 em data/raw. Os arquivos de 2018 a 2022 cont?m as 18 colunas por nome. O CSV de 2017 tem 45 campos e vem sem cabeçalho. Os tamanhos e hashes estão na [proveniência](data-provenance.md). Fora dos EUA, os painéis britânicos da FCA descrevem vendas de hipotecas, e a série usada em pesquisa do Banco da Inglaterra inclui contratos concluídos. [Painéis da FCA](https://www.fca.org.uk/data/product-sales-data/mortgages-dashboards) · [Estudo do Banco da Inglaterra](https://www.bankofengland.co.uk/-/media/boe/files/working-paper/2023/useful-usable-and-used-buffer-usability-during-the-covid-19-crisis.pdf)

Na busca por um equivalente brasileiro ao HMDA, a base pública mais próxima foi a versão analítica do Minha Casa, Minha Vida Financiado (FGTS/FS). O dicionário informa contratos, renda familiar, data de nascimento, valor financiado, valor de compra, tipo de imóvel e município. A base publicada cobre contratos e não oferece o desfecho aprovado/recusado nem raça ou cor do solicitante. Os agregados do Censo 2022 permitem acrescentar contexto territorial por município. Uma junção territorial não recupera a cor ou raça individual. O SCR.data do Banco Central tem operações agregadas por modalidade, UF e faixa de renda. [Bases do MCMV](https://www.gov.br/cidades/pt-br/acesso-a-informacao/acoes-e-programas/habitacao/programa-minha-casa-minha-vida/bases-de-dados-do-programa-minha-casa-minha-vida) · [Censo 2022](https://www.ibge.gov.br/estatisticas/sociais/populacao/22827-censo-demografico-2022.html?edicao=41852&t=downloads) · [SCR.data](https://dadosabertos.bcb.gov.br/pt_BR/dataset/scr_data)

O artigo explica termos técnicos e siglas quando aparecem, incluindo os métodos, HMDA, ROC-AUC, PR-AUC, precisão, revocação, acurácia balanceada, limiar, FPR, FNR e pontos percentuais. As legendas explicam as abreviações das tabelas.

O log do Overleaf mencionava `main.tex`, enquanto o manuscrito local se chama `artigo.tex`. No projeto existente, selecionar `artigo.tex` como documento principal. Também foi criado o [ZIP para importar no Overleaf](../reports/local/artigo-overleaf.zip), com a cópia do manuscrito como `main.tex`, os estilos, a bibliografia e as três figuras na raiz. A integridade e o conteúdo do ZIP foram conferidos localmente.

As três figuras usadas no artigo estão em `docs/figuras-conclusao/`. Para compilar no Overleaf, os PNGs podem ser enviados para a pasta do `artigo.tex`, mantendo os nomes `desempenho-valid18.png`, `recusas-raca-renda-valid18.png` e `perfis-fixos-valid18.png`.

Em 26/11/2026, confirmamos a execução completa 70/20/10 pelos relatórios em `reports/local/full_valid18_702010`. O [artigo](../artigo.tex) passou a usar esse recorte de 2.910.642 pedidos com os 18 campos válidos. A conclusão responde às seis perguntas em uma tabela, e três figuras em `docs/figuras-conclusao` mostram desempenho, recusas por faixa de renda e cenários com perfis fixos. A estabilidade entre instituições e anos segue em aberto.

O Passo 6 das [notas do pitch](oque-eu-entendi.md) responde às seis perguntas com três gráficos em `docs/figuras-conclusao/`. O gráfico de recusas observadas usa os 291.065 pedidos da partição de teste atual e mostra os dois grupos de raça por faixa de renda. Os gráficos de desempenho e perfis fixos reproduzem números da execução anterior documentados no artigo. A execução atual terminou. A tabela de respostas e as três figuras atualizadas constam no artigo.

O [protocolo de pesquisa](perguntas-e-protocolo-hmda.md) está em texto corrido. As perguntas, formas de comparação e métricas anteriores foram preservadas, com a indicação de qual execução gerou os resultados.

O Passo 5 das notas do pitch foi ampliado em linguagem simples. Usa uma ficha fictícia para explicar pesos e categorias na regressão, um caminho real da árvore para explicar a floresta e versões da mesma ficha para distinguir as oito combinações de treinamento dos 12 cenários de perfis fixos.

As notas do pitch explicam a divisão estratificada com 100 pedidos fictícios, dos quais 90 aprovados e 10 recusados. O exemplo mostra como as pilhas de treino, validação e teste mantêm a proporção e explica que a semente 42 permite repetir o sorteio.

A pedido do usuário, atualizar este arquivo a cada resposta sobre o projeto, inclusive orientações. Registrar ações, decisões, verificações e pendências pertinentes. Vincular documentos de `docs/` que tragam mais detalhes.

O ambiente local é `.venv` com Python 3.12. No PowerShell, use `.\.venv\Scripts\Activate.ps1`. No CMD, use `.venv\Scripts\activate.bat`. As dependências estão em [requirements.txt](../requirements.txt). Para abrir os notebooks localmente, execute `python -m jupyterlab` com o ambiente ativo.

Para executar os notebooks 03 e 04, a primeira opção é o PC local (Ryzen 5 PRO 5655GE, cerca de 16 GB de RAM), onde os ZIPs já estão disponíveis. O treino atual usa CPU. No Colab, os caminhos atuais passam pelo Google Drive, e a leitura das bases pode acrescentar tempo. Ainda sem medição comparativa entre os ambientes.

No notebook 04, o ajuste usa `SGDClassifier` e `RandomForestClassifier` do scikit-learn, com os dados na RAM do sistema e processamento na CPU. Ativar a GPU no Colab por si só não muda onde essas operações rodam. Para medir ganho com GPU, seria preciso testar implementações que usem esse hardware e conferir os resultados com a mesma divisão de dados.

Parquet é um formato de arquivo para tabelas que armazena os dados por coluna, guarda os tipos dos campos e permite compactação. No projeto, o notebook 03 lê os CSVs brutos dentro dos ZIPs e grava um Parquet por ano com os pedidos elegíveis que passaram na validação dos 18 campos. O notebook 04 lê esses arquivos para treinar e avaliar os modelos.

Para transferir a base preparada entre computadores pelo Git, o .gitignore libera os Parquets anuais finais em `data/processed/hmda_multiano/`. Os ZIPs brutos, arquivos parciais e relatórios locais seguem ignorados. Na conferência, estavam disponíveis os Parquets de 2018, 2019 e 2020. Para rodar o notebook 04 com todos os anos em outro computador, leve também os demais Parquets depois de concluir o notebook 03.

O notebook 03 usa `zipfile.ZipFile` para abrir o CSV dentro de cada ZIP. O Python descompacta o conteúdo conforme lê, e o pandas entrega as linhas em blocos para aplicar os filtros e a validação. Assim, o preparo grava o Parquet anual sem criar uma cópia do CSV bruto no disco.
