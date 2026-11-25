# Proveniência do HMDA 2024

- Fonte oficial: [Snapshot National Loan-Level Dataset de 2024, FFIEC/CFPB](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2024).
- Edição: Loan/Application Records (LAR), ano de referência 2024, snapshot nacional com dados congelados em 19/05/2025. Selecionamos essa edição para fixar a versão usada no estudo.
- Download em 24/11/2026: [ZIP oficial da edição CSV](https://files.ffiec.cfpb.gov/static-data/snapshot/2024/2024_public_lar_csv.zip).
- Arquivo original baixado: `data/raw/2024_public_lar_csv.zip`, 664.242.987 bytes. O arquivo disponível atualmente é o [CSV extraído](../data/raw/2024_public_lar_csv/2024_public_lar_csv.csv). A pasta de dados brutos é ignorada pelo Git.
- SHA-256 calculado localmente: `9383890f38c2ae1ef46942e72e1a2a211aa0b979c0e46db5fd6d7eb8669f22c1`.
- O ZIP passou na checagem de integridade após o download. O CSV tem 4.625.405.352 bytes e 99 colunas.

## Recorte planejado

Selecionar compra de residência principal com `loan_purpose = 1` e `occupancy_type = 1`. Manter `action_taken` nos códigos 1, 2 e 3. Codificar os dois primeiros como aprovação e o terceiro como recusa. O cabeçalho inclui esses campos, `activity_year` e `tract_to_msa_income_percentage`.

## Validação local em 24/11/2026

Dos 12.229.298 registros lidos, todos eram de 2024. O filtro de compra manteve 6.553.105 registros e o de residência principal manteve 5.864.623. Após o filtro de desfecho, restaram 3.794.116 solicitações, sendo 3.236.602 aprovações (85,31%) e 557.514 recusas (14,69%). As contagens de `validacao_etapas.csv` e `validacao_alvo.csv` conferem com os metadados do Parquet final.

Registrar aqui as contagens por etapa, a distribuição do alvo, o tratamento de códigos de ausência, os campos selecionados, a semente e a estratégia de particionamento quando forem definidos. O protocolo está resumido no [contexto do projeto](contexto-projeto.md).
