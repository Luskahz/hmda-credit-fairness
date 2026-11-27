"""Preparo, treinamento e avaliação histórica dos snapshots públicos do HMDA."""

from collections import Counter
from itertools import product, zip_longest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    auc, balanced_accuracy_score, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from contar_linhas_completas import COLUMNS, NUMERIC, valid_fields


YEARS = tuple(range(2018, 2026))
SEED = 42
PARTITIONS = ("ajuste", "validacao", "teste")
DEMOGRAPHICS = ("derived_race", "derived_ethnicity")
TRACT_INCOME = "tract_to_msa_income_percentage"
FILTERS = ("loan_purpose", "occupancy_type", "action_taken")
FEATURES = tuple(name for name in COLUMNS if name not in FILTERS)
BASE = tuple(name for name in FEATURES if name not in (*DEMOGRAPHICS, TRACT_INCOME))
CONFIGS = {
    "A": BASE,
    "A+B": BASE + DEMOGRAPHICS,
    "A+C": BASE + (TRACT_INCOME,),
    "A+B+C": BASE + DEMOGRAPHICS + (TRACT_INCOME,),
}
BATCH_SIZE = 100_000
MODELS = ("Regressão logística incremental", "Floresta em blocos")
RACIAL_REFERENCE = "White"
RACIAL_FOCUS = "Black or African American"


def prepare_year(archive, output, year):
    """Valida um ZIP e escreve um Parquet anual com a partição de cada pedido."""
    archive = Path(archive)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".parquet.part")
    schema = pa.schema(
        [(name, pa.float64() if name in NUMERIC else pa.string()) for name in FEATURES]
        + [("approved", pa.int8()), ("particao", pa.string())],
        metadata={b"hmda_validated_fields": b"18", b"hmda_year": str(year).encode()},
    )
    rng = np.random.default_rng(SEED + year)
    counts = Counter({key: 0 for key in ("lidos", "elegiveis", "validos", "excluidos")})
    invalid = Counter({name: 0 for name in COLUMNS})
    split_counts = Counter({(part, label): 0 for part in PARTITIONS for label in (0, 1)})
    with ZipFile(archive) as source:
        member, = (item for item in source.namelist() if item.lower().endswith(".csv"))
        with source.open(member) as stream, pq.ParquetWriter(partial, schema, compression="zstd") as writer:
            for chunk in pd.read_csv(
                stream, usecols=COLUMNS, dtype="string", keep_default_na=False,
                chunksize=BATCH_SIZE,
            ):
                if not chunk["activity_year"].eq(str(year)).all():
                    raise ValueError(f"{archive.name} contém registros de outro ano.")
                counts["lidos"] += len(chunk)
                eligible = chunk.loc[
                    chunk["loan_purpose"].eq("1") & chunk["occupancy_type"].eq("1")
                    & chunk["action_taken"].isin(("1", "2", "3"))
                ]
                valid = valid_fields(eligible, year=year)
                invalid.update((~valid).sum().to_dict())
                keep = valid.all(axis=1)
                counts["elegiveis"] += len(eligible)
                counts["validos"] += int(keep.sum())
                counts["excluidos"] += int((~keep).sum())
                if not keep.any():
                    continue
                frame = eligible.loc[keep, list(FEATURES)].copy()
                for name in NUMERIC:
                    frame[name] = pd.to_numeric(frame[name], errors="raise")
                frame["approved"] = eligible.loc[keep, "action_taken"].isin(("1", "2")).astype("int8")
                # Separa cada classe dentro de cada ano. As proporções são aproximadas.
                partition = np.empty(len(frame), dtype="object")
                for label in (0, 1):
                    position = np.flatnonzero(frame["approved"].to_numpy() == label)
                    rng.shuffle(position)
                    fit_end = round(len(position) * 0.70)
                    val_end = round(len(position) * 0.90)
                    for name, indices in zip(
                        PARTITIONS, (position[:fit_end], position[fit_end:val_end], position[val_end:]),
                        strict=True,
                    ):
                        partition[indices] = name
                        split_counts[name, label] += len(indices)
                frame["particao"] = partition
                for name in PARTITIONS:
                    subset = frame.loc[frame["particao"].eq(name)]
                    if not subset.empty:
                        writer.write_table(pa.Table.from_pandas(subset, schema=schema, preserve_index=False))
    with pq.ParquetFile(partial) as parquet:
        if parquet.metadata.num_rows != counts["validos"] or not all(split_counts.values()):
            raise ValueError(f"Partições incompletas no ano {year}.")
    partial.replace(output)
    partitions = [
        {"ano": year, "particao": part, "n": split_counts[part, 0] + split_counts[part, 1],
         "aprovados": split_counts[part, 1], "recusados": split_counts[part, 0]}
        for part in PARTITIONS
    ]
    return {"ano": year, **counts}, partitions, [
        {"ano": year, "campo": name, "invalidos": count} for name, count in invalid.items()
    ]


def annual_datasets(folder, years):
    datasets = {}
    for year in years:
        path = Path(folder) / f"hmda_{year}_valid18.parquet"
        with pq.ParquetFile(path) as file:
            metadata = file.schema_arrow.metadata or {}
            if metadata.get(b"hmda_validated_fields") != b"18" or metadata.get(b"hmda_year") != str(year).encode():
                raise ValueError(f"Arquivo incompatível com o preparo anual: {path}")
        datasets[year] = ds.dataset(path, format="parquet")
    return datasets


def batches(dataset, part, columns, size=BATCH_SIZE):
    scanner = dataset.scanner(columns=list(columns), filter=ds.field("particao") == part, batch_size=size)
    for batch in scanner.to_batches():
        if batch.num_rows:
            yield batch.to_pandas()


def fit_preprocessing(datasets):
    scaler = StandardScaler()
    categorical = [name for name in FEATURES if name not in NUMERIC]
    categories = {name: set() for name in categorical}
    for dataset in datasets.values():
        for frame in batches(dataset, "ajuste", FEATURES):
            scaler.partial_fit(frame[list(NUMERIC)])
            for name in categorical:
                categories[name].update(frame[name].unique())
    vocab = {name: sorted(values) for name, values in categories.items()}
    return scaler, vocab


def encoders_for(names, vocab):
    categorical = [name for name in names if name in vocab]
    categories = [vocab[name] for name in categorical]
    encoder = OneHotEncoder(categories=categories, handle_unknown="ignore", dtype=np.float32)
    encoder.fit(pd.DataFrame({name: [vocab[name][0]] for name in categorical}))
    return categorical, encoder


def matrices(frame, names, scaler, categorical, encoder):
    numeric = [name for name in NUMERIC if name in names]
    encoded = encoder.transform(frame[categorical])
    indices = [NUMERIC.index(name) for name in numeric]
    raw = frame[numeric].to_numpy(dtype=np.float32)
    scaled = ((raw - scaler.mean_[indices]) / scaler.scale_[indices]).astype(np.float32)
    return hstack((csr_matrix(scaled), encoded), format="csr"), hstack(
        (csr_matrix(raw), encoded), format="csr"
    )


def fit_models(datasets, names, scaler, vocab, train_counts, target_trees=100):
    """Usa todos os pedidos em SGD e distribui o ajuste da floresta em blocos multiano."""
    categorical, encoder = encoders_for(names, vocab)
    columns = (*names, "approved")
    logistic = SGDClassifier(loss="log_loss", average=True, random_state=SEED)
    streams = [
        iter(batches(dataset, "ajuste", columns, size=max(1, int(np.ceil(train_counts[year] / target_trees)))))
        for year, dataset in datasets.items()
    ]
    # Cada árvore recebe pedidos de vários anos. Fragmentos de row groups
    # podem criar blocos extras. Coalescemos até o tamanho desejado por ano.
    def coalesce(stream, limit):
        pending = []
        length = 0
        for frame in stream:
            offset = 0
            while offset < len(frame):
                amount = min(limit - length, len(frame) - offset)
                pending.append(frame.iloc[offset:offset + amount])
                length += amount
                offset += amount
                if length == limit:
                    yield pd.concat(pending, ignore_index=True)
                    pending, length = [], 0
        if pending:
            yield pd.concat(pending, ignore_index=True)

    streams = [
        coalesce(stream, max(1, int(np.ceil(train_counts[year] / target_trees))))
        for year, stream in zip(datasets, streams, strict=True)
    ]
    forests = []
    n_seen = 0
    for frames in zip_longest(*streams):
        block = pd.concat([frame for frame in frames if frame is not None], ignore_index=True)
        order = np.random.default_rng(SEED + len(forests)).permutation(len(block))
        block = block.iloc[order]
        y = block["approved"].to_numpy()
        x_logistic, x_tree = matrices(block, names, scaler, categorical, encoder)
        logistic.partial_fit(x_logistic, y, classes=[0, 1])
        forest = RandomForestClassifier(
            n_estimators=1, max_depth=20, min_samples_leaf=5,
            random_state=SEED + len(forests),
        )
        forest.fit(x_tree, y)
        forests.append(forest)
        n_seen += len(block)
    if n_seen != sum(train_counts.values()):
        raise ValueError("O número de pedidos processados diverge da partição de ajuste.")
    return logistic, forests, categorical, encoder


def probabilities(frame, names, scaler, fitted):
    logistic, forests, categorical, encoder = fitted
    x_logistic, x_tree = matrices(frame, names, scaler, categorical, encoder)
    p_forest = np.zeros(len(frame))
    for forest in forests:
        if 1 in forest.classes_:
            index = np.flatnonzero(forest.classes_ == 1)[0]
            p_forest += forest.predict_proba(x_tree)[:, index]
    return {MODELS[0]: logistic.predict_proba(x_logistic)[:, 1],
            MODELS[1]: p_forest / len(forests)}


def predict_batches(dataset, part, names, scaler, fitted, audit=False):
    audit_fields = (*DEMOGRAPHICS, TRACT_INCOME) if audit else ()
    fields = dict.fromkeys((*names, "approved", *audit_fields))
    chunks = []
    for frame in batches(dataset, part, fields):
        result = {"approved": frame["approved"].to_numpy(),
                  **probabilities(frame, names, scaler, fitted)}
        if audit:
            result.update({name: frame[name].to_numpy() for name in DEMOGRAPHICS})
            band = pd.cut(frame[TRACT_INCOME], [-np.inf, 50, 80, 120, np.inf],
                          labels=["<50%", "50 a <80%", "80 a <120%", "120% ou mais"], right=False)
            result["renda_setor"] = band.astype(str).to_numpy()
            result["raca_x_renda"] = frame["derived_race"].str.cat(band.astype(str), sep=" | ").to_numpy()
            result["etnia_x_renda"] = frame["derived_ethnicity"].str.cat(band.astype(str), sep=" | ").to_numpy()
        chunks.append(pd.DataFrame(result))
    return pd.concat(chunks, ignore_index=True)


def evaluate_scores(observed, scores, threshold):
    predicted = scores >= threshold
    precision, recall, _ = precision_recall_curve(1 - observed, 1 - scores)
    return {
        "n": len(observed), "roc_auc": roc_auc_score(observed, scores),
        "pr_auc_recusa": auc(recall, precision),
        "acuracia_balanceada": balanced_accuracy_score(observed, predicted),
        "recusa_observada": (observed == 0).mean(), "recusa_prevista": (~predicted).mean(),
    }


def group_rates(frame, field, score_field, threshold):
    groups = []
    for category, subset in frame.groupby(field, sort=True):
        accepted = subset["approved"].eq(1).to_numpy()
        predicted = subset[score_field].ge(threshold).to_numpy()
        groups.append({
            "dimensao": field, "grupo": category, "n": len(subset),
            "n_aprovados": int(accepted.sum()), "n_recusados": int((~accepted).sum()),
            "recusa_observada": (~accepted).mean(), "recusa_prevista": (~predicted).mean(),
            "fpr": predicted[~accepted].mean() if (~accepted).any() else np.nan,
            "fnr": (~predicted[accepted]).mean() if accepted.any() else np.nan,
        })
    return groups


def fixed_profiles(dataset, year, cutoff, names, scaler, fitted):
    """Repete uma amostra reprodutível de perfis de teste em 12 cenários."""
    total = dataset.count_rows(filter=ds.field("particao") == "teste")
    rng = np.random.default_rng(SEED + year)
    positions = np.sort(rng.choice(total, size=min(2500, total), replace=False))
    selected = []
    offset = 0
    for frame in batches(dataset, "teste", names):
        low, high = np.searchsorted(positions, [offset, offset + len(frame)])
        if high > low:
            selected.append(frame.iloc[positions[low:high] - offset])
        offset += len(frame)
    profiles = pd.concat(selected, ignore_index=True)
    rows = []
    for race, ethnicity, income in product(
        (RACIAL_REFERENCE, RACIAL_FOCUS, "American Indian or Alaska Native"),
        ("Not Hispanic or Latino", "Hispanic or Latino"), (67.0, 143.0),
    ):
        scenario = profiles.assign(derived_race=race, derived_ethnicity=ethnicity,
                                   **{TRACT_INCOME: income})
        for model, values in probabilities(scenario, names, scaler, fitted).items():
            rows.append({"ano": year, "algoritmo": model, "raca": race, "etnia": ethnicity,
                         "renda_relativa": income, "n_perfis": len(profiles),
                         "prob_media_aprovacao": values.mean(),
                         "taxa_aprovacao_prevista": (values >= cutoff[model]).mean()})
    return rows


def train_and_compare(datasets, partitions, target_trees=100):
    """Escolhe limiares na validação conjunta e compara no teste total e por ano."""
    for year, dataset in datasets.items():
        expected = partitions.loc[partitions.ano.eq(year)].set_index("particao")["n"]
        if set(expected.index) != set(PARTITIONS) or not expected.index.is_unique:
            raise ValueError(f"As partições do ano {year} estão incompletas.")
        for part in PARTITIONS:
            if dataset.count_rows(filter=ds.field("particao") == part) != expected[part]:
                raise ValueError(f"O Parquet e a contagem de {part} divergem em {year}. Refaça o preparo.")
    scaler, vocab = fit_preprocessing(datasets)
    train_counts = partitions.loc[partitions.particao.eq("ajuste")].set_index("ano")["n"].to_dict()
    metrics, groups, scenarios = [], [], []
    for config, names in CONFIGS.items():
        print(f"Ajustando configuração {config}", flush=True)
        fitted = fit_models(datasets, names, scaler, vocab, train_counts, target_trees)
        validation = pd.concat(
            [predict_batches(dataset, "validacao", names, scaler, fitted) for dataset in datasets.values()],
            ignore_index=True,
        )
        cutoffs = {}
        for model in MODELS:
            fpr, tpr, thresholds = roc_curve(validation["approved"], validation[model])
            cutoffs[model] = thresholds[np.argmax(np.where(np.isfinite(thresholds), tpr - fpr, -np.inf))]
        del validation
        test = {
            year: predict_batches(dataset, "teste", names, scaler, fitted, audit=True)
            for year, dataset in datasets.items()
        }
        test["conjunto"] = pd.concat(list(test.values()), ignore_index=True)
        for model, cutoff in cutoffs.items():
            for year, frame in test.items():
                keys = {"ano": year, "algoritmo": model, "configuracao": config, "limiar": cutoff}
                metrics.append({**keys, **evaluate_scores(frame["approved"].to_numpy(), frame[model], cutoff),
                                "arvores": len(fitted[1]) if model == "Floresta em blocos" else 0})
                for field in (*DEMOGRAPHICS, "renda_setor", "raca_x_renda", "etnia_x_renda"):
                    groups.extend({**keys, **row} for row in group_rates(frame, field, model, cutoff))
        if config == "A+B+C":
            for year, dataset in datasets.items():
                scenarios.extend(fixed_profiles(dataset, year, cutoffs, names, scaler, fitted))
        del test, frame, fitted
    return pd.DataFrame(metrics), pd.DataFrame(groups), pd.DataFrame(scenarios)


def disparity_trends(groups):
    comparisons = [
        ("derived_race", RACIAL_FOCUS, RACIAL_REFERENCE),
        ("derived_ethnicity", "Hispanic or Latino", "Not Hispanic or Latino"),
    ]
    keys = ["ano", "algoritmo", "configuracao"]
    rates = ["recusa_observada", "recusa_prevista", "fpr", "fnr"]
    outputs = []
    for field, target, reference in comparisons:
        subset = groups.loc[groups.dimensao.eq(field)]
        first = subset.loc[subset.grupo.eq(target), keys + ["n", *rates]]
        second = subset.loc[subset.grupo.eq(reference), keys + ["n", *rates]]
        result = first.merge(second, on=keys, suffixes=("_grupo", "_ref"), validate="one_to_one")
        result["dimensao"], result["grupo"], result["referencia"] = field, target, reference
        for rate in rates:
            result[f"delta_{rate}_pp"] = 100 * (result[f"{rate}_grupo"] - result[f"{rate}_ref"])
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True)
