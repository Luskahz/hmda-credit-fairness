# HMDA Credit Fairness

Projeto final da disciplina de Ciência de Dados (CARCDSI), desenvolvido por Lucas Alves de Melo Lins e João Gabriel de Faria Beserra.

## Objetivo

Investigar como a inclusão de raça, etnia e renda relativa da região do imóvel afeta previsões de **aprovação ou recusa** de financiamentos imobiliários, utilizando decisões históricas do Home Mortgage Disclosure Act (HMDA).

O alvo será derivado de `action_taken`: **1** para códigos 1 e 2 (aprovação) e **0** para código 3 (recusa), após definição e documentação dos filtros de elegibilidade. Não se trata de previsão de inadimplência nem de recomendação de concessão real de crédito.

## Desenho experimental previsto

- **A:** atributos financeiros, financiamento, imóvel, instituição e período;
- **A+B:** A mais raça e etnia;
- **A+C:** A mais renda relativa da região do imóvel;
- **A+B+C:** combinação completa.

Cada configuração será avaliada com Regressão Logística e Random Forest, com mesma amostra e partições. O protocolo definitivo será estabelecido após inspeção do dataset.

## Ambiente local

Requer Python **3.12**.

```bash
git clone https://github.com/Luskahz/hmda-credit-fairness.git
cd hmda-credit-fairness
python -m venv .venv
```

Ative o ambiente:

- Windows PowerShell: `\.venv\Scripts\Activate.ps1`
- Linux/macOS: `source .venv/bin/activate`

Instale o pacote e as ferramentas de desenvolvimento:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

A instalação não baixa os dados e não treina modelos.

## Organização

- `src/hmda_credit_fairness/`: módulos reutilizáveis de dados, atributos, modelos e análises;
- `notebooks/`: exploração e visualizações;
- `data/raw/`: arquivos originais locais;
- `data/interim/`: recortes intermediários locais;
- `data/processed/`: dados preparados locais;
- `reports/figures/`: figuras selecionadas para publicação;
- `tests/`: testes automatizados.

## Dataset e reprodutibilidade

**O dataset ainda não foi escolhido/baixado.** Não commitar arquivos brutos, intermediários ou processados no Git. Na próxima etapa, serão documentados fonte oficial, ano, versão, data de obtenção, filtros, contagens, integridade e estratégia de amostragem. O código de download e preparação será implementado somente após essas decisões.

## Status

Estrutura inicial: sem pipelines de treinamento ou resultados científicos implementados.
