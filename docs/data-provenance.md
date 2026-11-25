# Proveniência do HMDA 2024

- Fonte oficial: [Snapshot National Loan-Level Dataset de 2024, FFIEC/CFPB](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2024).
- Edição: Loan/Application Records (LAR), ano de referência 2024, snapshot nacional com dados congelados em 19/05/2025. Selecionamos essa edição para fixar a versão usada no estudo.
- Download em 24/11/2026: [ZIP oficial da edição CSV](https://files.ffiec.cfpb.gov/static-data/snapshot/2024/2024_public_lar_csv.zip).
- Arquivo local: [data/raw/2024_public_lar_csv.zip](../data/raw/2024_public_lar_csv.zip), 664.242.987 bytes. A pasta de dados brutos é ignorada pelo Git.
- SHA-256 calculado localmente: `9383890f38c2ae1ef46942e72e1a2a211aa0b979c0e46db5fd6d7eb8669f22c1`.
- O ZIP passou na checagem de integridade. Contém `2024_public_lar_csv.csv`, com 4.625.405.352 bytes descompactado e 99 colunas. A distribuição de origem foi preservada compactada.

## Recorte planejado

Selecionar compra de residência principal com `loan_purpose = 1` e `occupancy_type = 1`. Manter `action_taken` nos códigos 1, 2 e 3. Codificar os dois primeiros como aprovação e o terceiro como recusa. O cabeçalho inclui esses campos, `activity_year` e `tract_to_msa_income_percentage`. Os filtros ainda não foram executados.

Registrar aqui as contagens por etapa, a distribuição do alvo, o tratamento de códigos de ausência, os campos selecionados, a semente e a estratégia de particionamento quando forem definidos. O protocolo está resumido no [contexto do projeto](contexto-projeto.md).
