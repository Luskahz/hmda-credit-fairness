"""Conta registros do HMDA 2024 com as 18 colunas usadas no preparo."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CSV = (
    Path(__file__).resolve().parents[1]
    / "data/raw/2024_public_lar_csv/2024_public_lar_csv.csv"
)
NUMERIC = ("income", "loan_amount", "loan_term", "property_value",
           "tract_to_msa_income_percentage")
COLUMNS = NUMERIC + (
    "lei", "activity_year", "applicant_age", "debt_to_income_ratio",
    "loan_type", "lien_status", "construction_method", "total_units",
    "derived_race", "derived_ethnicity", "loan_purpose", "occupancy_type",
    "action_taken",
)
# Categorias informativas do LAR público. Códigos como 8888 (idade
# indisponível), Exempt e Free Form Text Only não informam o atributo.
CATEGORIES = {
    "activity_year": ("2024",),
    "applicant_age": ("<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"),
    "debt_to_income_ratio": (
        "<20%", "20%-<30%", "30%-<36%", "50%-60%", ">60%",
        # O CSV de 2024 omite o sinal de porcentagem nos valores inteiros.
        *(str(value) for value in range(36, 50)),
        *(f"{value}%" for value in range(36, 50)),
    ),
    "loan_type": ("1", "2", "3", "4"),
    "lien_status": ("1", "2"),
    "construction_method": ("1", "2"),
    "total_units": ("1", "2", "3", "4", "5-24", "25-49", "50-99", "100-149", ">149"),
    "derived_race": (
        "American Indian or Alaska Native", "Asian", "Black or African American",
        "Native Hawaiian or Other Pacific Islander", "White", "2 or more minority races", "Joint",
    ),
    "derived_ethnicity": ("Hispanic or Latino", "Not Hispanic or Latino", "Joint"),
    "loan_purpose": ("1", "2", "31", "32", "4"),
    "occupancy_type": ("1", "2", "3"),
    "action_taken": ("1", "2", "3", "4", "5", "6", "7", "8"),
}


def valid_fields(chunk, *, year=2024):
    valid = {name: chunk[name].isin(values).to_numpy(dtype=bool)
             for name, values in CATEGORIES.items()}
    valid["activity_year"] = chunk["activity_year"].eq(str(year)).to_numpy(dtype=bool)
    valid["lei"] = chunk["lei"].str.fullmatch(r"[A-Z0-9]{20}").to_numpy(dtype=bool)
    for name in NUMERIC:
        numeric = pd.to_numeric(chunk[name], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(numeric)
        if name != "income":
            mask &= numeric > 0
        if name == "loan_term":
            mask &= numeric == np.floor(numeric)
        valid[name] = mask
    return pd.DataFrame(valid, index=chunk.index, columns=COLUMNS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f"CSV não encontrado: {args.csv}")

    totals = np.zeros((2, 2), dtype=np.int64)
    invalid_eligible = pd.Series(0, index=COLUMNS, dtype="int64")
    for chunk in pd.read_csv(
        args.csv, usecols=COLUMNS, dtype="string", keep_default_na=False,
        chunksize=100_000,
    ):
        eligible = (
            chunk["activity_year"].eq("2024")
            & chunk["loan_purpose"].eq("1")
            & chunk["occupancy_type"].eq("1")
            & chunk["action_taken"].isin(["1", "2", "3"])
        ).to_numpy(dtype=bool)
        valid = valid_fields(chunk)
        complete = valid.all(axis=1).to_numpy(dtype=bool)
        totals[0] += (len(chunk), complete.sum())
        totals[1] += (eligible.sum(), (eligible & complete).sum())
        invalid_eligible += (~valid.loc[eligible]).sum()

    for label, (rows, complete) in zip(
        ("CSV completo", "Recorte elegível"), totals, strict=True
    ):
        print(f"\n{label}")
        print(f"  Total de linhas: {rows:,}")
        print(f"  Linhas com os 18 campos válidos: {complete:,}")
        if rows:
            print(f"  Proporção com os 18 válidos: {complete / rows:.2%}")

    print("\nCampos inválidos ou indisponíveis no recorte elegível (contagens por coluna)")
    for name, count in invalid_eligible.sort_values(ascending=False).items():
        print(f"  {name}: {count:,}")


if __name__ == "__main__":
    main()
