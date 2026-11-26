# Proveniência do HMDA 2024

## Snapshots de 2023 a 2025

Em 26/11/2026, salvamos os três ZIPs oficiais em `data/raw`. Todos passaram na verificação CRC e os cabeçalhos têm 99 colunas, incluindo as 18 usadas no estudo. O hash de 2024 também coincide com o download anterior. [Portal oficial dos snapshots](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/)

| Ano | Arquivo local | Bytes do ZIP | SHA-256 local |
| --- | --- | ---: | --- |
| 2023 | [2023_public_lar_csv.zip](../data/raw/2023_public_lar_csv.zip) | 624.535.331 | `addc78ab14d89ba5ea3169386c514421459f2d59978885c6dbdf54d2a567b19a` |
| 2024 | [2024_public_lar_csv.zip](../data/raw/2024_public_lar_csv.zip) | 664.242.987 | `9383890f38c2ae1ef46942e72e1a2a211aa0b979c0e46db5fd6d7eb8669f22c1` |
| 2025 | [2025_public_lar_csv.zip](../data/raw/2025_public_lar_csv.zip) | 737.139.477 | `9b25c05d15704a49889e585a930e0273fecbe3f3eade44cb6ee773d6ab628ff2` |

## Snapshots históricos de 2017 a 2022

Os seis ZIPs nacionais LAR foram baixados em 26/11/2026 para `data/raw`. Conferimos a integridade CRC de todos os arquivos e calculamos o SHA-256 local. Os CSVs continuam dentro dos ZIPs. [Portal oficial dos snapshots](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/)

| Ano | Arquivo local | Bytes do ZIP | Colunas | SHA-256 local |
| --- | --- | ---: | ---: | --- |
| 2017 | [2017_public_lar_csv.zip](../data/raw/2017_public_lar_csv.zip) | 155.879.359 | 45 | `186b977af62198a3e689c99cb3559fddf3a0937e0a752e0c1f20c61c79ca15c5` |
| 2018 | [2018_public_lar_csv.zip](../data/raw/2018_public_lar_csv.zip) | 823.719.647 | 99 | `a165341cc53205f4069ae50994b5adff4c679ffc9389a94d483154c70c8f67ce` |
| 2019 | [2019_public_lar_csv.zip](../data/raw/2019_public_lar_csv.zip) | 980.129.414 | 99 | `f1120b4a331e80e85a904a5c548309e3d3b9a5ce6882257ed7437b30bb4faad9` |
| 2020 | [2020_public_lar_csv.zip](../data/raw/2020_public_lar_csv.zip) | 1.460.346.740 | 99 | `3bd5c8d21f37a25ae5701e627a19f4cb7e81ca02c4b209184be6f7b07c6ea5d9` |
| 2021 | [2021_public_lar_csv.zip](../data/raw/2021_public_lar_csv.zip) | 1.517.879.241 | 99 | `20d61f6fc2e1e9c90b9199bb9075602cda718b4aa905807a6d8dec8433c7a3d3` |
| 2022 | [2022_public_lar_csv.zip](../data/raw/2022_public_lar_csv.zip) | 877.742.261 | 99 | `df7ad5b544fe8dc7ca05de58a1975fd8970b829136f1538cea83882773e759d4` |

Os CSVs de 2018 a 2022 têm as 18 colunas usadas no estudo com os mesmos nomes. A validação dos valores e dos códigos por ano ainda precisa ser feita antes de treinar. O CSV de 2017 tem 45 campos por registro e vem sem cabeçalho. Ele exige um preparo separado com o layout oficial. [Layout do LAR 2017](https://files.ffiec.cfpb.gov/static-data/snapshot/2017/2017_publicstatic_dataformat.pdf)

- Fonte oficial: [Snapshot National Loan-Level Dataset de 2024, FFIEC/CFPB](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2024).
- Edição: Loan/Application Records (LAR), ano de referência 2024, snapshot nacional com dados congelados em 19/05/2025. Selecionamos essa edição para fixar a versão usada no estudo.
- Download em 24/11/2026: [ZIP oficial da edição CSV](https://files.ffiec.cfpb.gov/static-data/snapshot/2024/2024_public_lar_csv.zip).
- Em 25/11/2026, baixamos novamente o ZIP oficial. O SHA-256 conferiu com o download anterior, e o CSV foi extraído no caminho usado pelo notebook 01.
- Arquivo original baixado: `data/raw/2024_public_lar_csv.zip`, 664.242.987 bytes. O arquivo disponível atualmente é o [CSV extraído](../data/raw/2024_public_lar_csv/2024_public_lar_csv.csv). A pasta de dados brutos é ignorada pelo Git.
- SHA-256 calculado localmente: `9383890f38c2ae1ef46942e72e1a2a211aa0b979c0e46db5fd6d7eb8669f22c1`.
- O ZIP passou na checagem de integridade após o download. O CSV tem 4.625.405.352 bytes e 99 colunas.

## Recorte planejado

Selecionar compra de residência principal com `loan_purpose = 1` e `occupancy_type = 1`. Manter `action_taken` nos códigos 1, 2 e 3. Codificar os dois primeiros como aprovação e o terceiro como recusa. O cabeçalho inclui esses campos, `activity_year` e `tract_to_msa_income_percentage`.

## Validação local em 24/11/2026

Dos 12.229.298 registros lidos, todos eram de 2024. O filtro de compra manteve 6.553.105 registros e o de residência principal manteve 5.864.623. Após o filtro de desfecho, restaram 3.794.116 solicitações, sendo 3.236.602 aprovações (85,31%) e 557.514 recusas (14,69%). As contagens de `validacao_etapas.csv` e `validacao_alvo.csv` conferem com os metadados do Parquet final.

A [auditoria das 18 colunas](../scripts/contar_linhas_completas.py) com regras por campo encontrou 5.998.031 linhas válidas no CSV bruto (49,05%) e 2.910.642 no recorte elegível (76,71%). A auditoria aceita categorias informativas, verifica o formato do LEI e aplica regras numéricas, incluindo prazo inteiro positivo. Marcadores como idade `8888` e categorias demográficas sem classificação, incluindo `Free Form Text Only`, ficam fora da contagem. No recorte, 883.474 linhas têm ao menos um campo inválido ou indisponível segundo essas regras. [Dicionário público do HMDA](https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields)

O preparo atualizado gravou as 2.910.642 linhas válidas em `data/processed/valid_18/hmda_2024_purchase_valid18.parquet`. São 2.570.870 aprovações e 339.772 recusas. Os CSVs da mesma pasta documentam as etapas e a quantidade de valores inválidos por campo.

O notebook 02 divide esse Parquet em 2.037.448 pedidos de treinamento, 582.129 de validação e 291.065 de teste. A divisão é estratificada pelo desfecho, com proporções aproximadas de 70%, 20% e 10% e semente 42.

Registrar aqui as contagens por etapa, a distribuição do alvo, o tratamento de códigos de ausência, os campos selecionados, a semente e a estratégia de particionamento quando forem definidos. O protocolo está resumido no [contexto do projeto](contexto-projeto.md).
