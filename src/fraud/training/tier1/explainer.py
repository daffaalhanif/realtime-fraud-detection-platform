"""Pengelompokan fitur grup V yang saling berkorelasi tinggi untuk pelaporan kontribusi SHAP.

Kontribusi SHAP pada fitur yang saling berkorelasi tinggi bisa terbagi di antara mereka, sehingga
masing-masing tampak kecil padahal gabungannya besar. Penjelasan untuk analis karena itu
menjumlahkan kontribusi satu kelompok dan menampilkannya sebagai satu baris.

Kelompok dibentuk dengan complete linkage atas korelasi absolut di split train, sehingga setiap
pasangan anggota satu kelompok berkorelasi paling tidak sebesar ambang. Ambang dipilih manusia
dari tabel sensitivitas `--report-only`. Kolom V adalah kolom mentah transaksi, jadi kelompoknya
berlaku untuk semua versi tier 1 yang memakai kolom tersebut.

Dijalankan lewat:
    uv run python -m fraud.training.tier1.explainer --report-only [--smoke]
    uv run python -m fraud.training.tier1.explainer --correlation-threshold T [--smoke]
"""

import argparse
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from fraud.features.encoding import FeatureSpec
from fraud.training.tier1.candidates import SMOKE_ROW_LIMIT
from fraud.training.tier1.dataset import prepare_datasets

FEATURE_GROUPS_PATH = Path("configs/feature_groups.json")

GROUPED_COLUMN_PATTERN = re.compile(r"V(\d+)")

# Galat baku korelasi sekitar 1/sqrt(n): pasangan yang jarang terisi bersamaan memberi korelasi
# yang terlalu berisik untuk dasar penggabungan, jadi diperlakukan tidak berkorelasi.
MIN_OVERLAP_ROWS = 1_000

REPORT_THRESHOLDS = (0.80, 0.90, 0.95)

# Pasangan V berkorelasi tinggi yang tercatat saat eksplorasi data awal, sebagai titik periksa.
REFERENCE_PAIRS = (("V244", "V242"), ("V201", "V200"), ("V257", "V246"), ("V189", "V188"))

# Fitur V dengan korelasi tertinggi ke label adalah kasus yang memicu pelaporan berkelompok.
TOP_TARGET_CORRELATED = 10


def _column_number(column: str) -> int:
    match = GROUPED_COLUMN_PATTERN.fullmatch(column)
    assert match is not None
    return int(match.group(1))


def grouped_family_columns(spec: FeatureSpec) -> list[str]:
    """Kolom grup V di kontrak input, urut nomor kolom."""
    columns = [c for c in spec["input_columns"] if GROUPED_COLUMN_PATTERN.fullmatch(c)]
    return sorted(columns, key=_column_number)


def absolute_correlation(features: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Korelasi Pearson absolut antar kolom, dihitung dari baris yang terisi pada kedua kolom.

    Returns:
        Matriks simetris; NaN untuk pasangan yang terisi bersamaan kurang dari
        `MIN_OVERLAP_ROWS` baris atau salah satu kolomnya konstan.
    """
    values = features.loc[:, columns].astype(np.float64)
    return values.corr(method="pearson", min_periods=MIN_OVERLAP_ROWS).abs()


def correlated_groups(abs_corr: pd.DataFrame, threshold: float) -> list[list[str]]:
    """Kelompok kolom yang setiap pasangan anggotanya berkorelasi absolut minimal `threshold`.

    Returns:
        Kelompok beranggota dua kolom atau lebih, anggota dan kelompok urut nomor kolom. Kolom
        yang tidak masuk kelompok mana pun tidak dicantumkan.
    """
    distance = 1.0 - abs_corr.fillna(0.0).to_numpy()
    np.fill_diagonal(distance, 0.0)
    # Pembulatan float bisa memberi korelasi sedikit di atas 1, padahal jarak tidak boleh negatif.
    distance = np.clip(distance, 0.0, None)
    tree = linkage(squareform(distance, checks=False), method="complete")
    labels = fcluster(tree, t=1.0 - threshold, criterion="distance")

    members: dict[int, list[str]] = {}
    for column, label in zip(abs_corr.columns, labels, strict=True):
        members.setdefault(int(label), []).append(str(column))
    groups = [sorted(group, key=_column_number) for group in members.values() if len(group) > 1]
    return sorted(groups, key=lambda group: _column_number(group[0]))


def top_target_correlated(
    features: pd.DataFrame, label: pd.Series, columns: list[str]
) -> list[str]:
    """Kolom dengan korelasi absolut tertinggi terhadap label."""
    # Kolom konstan memberi pembagian nol dan korelasi NaN; itu wajar, bukan galat.
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = (
            features.loc[:, columns].astype(np.float64).corrwith(label.astype(np.float64)).abs()
        )
    return list(correlation.nlargest(TOP_TARGET_CORRELATED).index)


def sensitivity_table(
    abs_corr: pd.DataFrame, top_columns: list[str], thresholds: tuple[float, ...]
) -> pd.DataFrame:
    """Ringkasan hasil pengelompokan pada beberapa ambang, bahan memilih ambang.

    Kolom `top_units` adalah jumlah baris laporan (kelompok atau kolom tunggal) yang ditempati
    kolom-kolom berkorelasi tertinggi ke label; makin kecil, makin sedikit kontribusinya terpecah.
    """
    rows = []
    for threshold in thresholds:
        groups = correlated_groups(abs_corr, threshold)
        group_of = {column: index for index, group in enumerate(groups) for column in group}
        units = {group_of.get(column, column) for column in top_columns}
        pairs_together = sum(
            a in group_of and group_of[a] == group_of.get(b) for a, b in REFERENCE_PAIRS
        )
        rows.append(
            {
                "threshold": threshold,
                "n_groups": len(groups),
                "grouped_columns": len(group_of),
                "largest_group": max((len(group) for group in groups), default=0),
                "reference_pairs_together": f"{pairs_together}/{len(REFERENCE_PAIRS)}",
                "top_units": f"{len(units)}/{len(top_columns)}",
            }
        )
    return pd.DataFrame(rows)


def groups_document(groups: list[list[str]], threshold: float, n_train_rows: int) -> dict:
    """Isi `configs/feature_groups.json` yang dibaca consumer narasi."""
    return {
        "method": "complete_linkage_abs_pearson",
        "correlation_threshold": threshold,
        "min_overlap_rows": MIN_OVERLAP_ROWS,
        "source_split": "train",
        "n_train_rows": n_train_rows,
        "groups": [
            {"id": f"V-{index}", "members": members}
            for index, members in enumerate(groups, start=1)
        ],
    }


def run_grouping(threshold: float | None, smoke: bool, output: Path) -> None:
    """Mencetak tabel sensitivitas, atau menulis kelompok pada satu ambang ke `output`.

    Args:
        threshold: Ambang korelasi; None berarti hanya mencetak tabel sensitivitas.
        smoke: True untuk memotong split train, hanya untuk uji coba cepat.
        output: Lokasi berkas kelompok yang ditulis.
    """
    data = prepare_datasets()
    features, label = data.train.features, data.train.label
    if smoke:
        features, label = features.iloc[:SMOKE_ROW_LIMIT], label.iloc[:SMOKE_ROW_LIMIT]
    columns = grouped_family_columns(data.spec)
    abs_corr = absolute_correlation(features, columns)

    if threshold is None:
        top_columns = top_target_correlated(features, label, columns)
        print(f"{len(columns)} kolom V, {len(features)} baris train")
        print(f"{TOP_TARGET_CORRELATED} kolom V berkorelasi tertinggi ke label: {top_columns}")
        print(sensitivity_table(abs_corr, top_columns, REPORT_THRESHOLDS).to_string(index=False))
        return

    document = groups_document(correlated_groups(abs_corr, threshold), threshold, len(features))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n")
    print(f"{len(document['groups'])} kelompok V pada ambang {threshold} ditulis ke {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pengelompokan fitur V berkorelasi tinggi.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--report-only",
        action="store_true",
        help="Cetak tabel sensitivitas ambang korelasi tanpa menulis berkas.",
    )
    mode.add_argument(
        "--correlation-threshold",
        type=float,
        help="Ambang korelasi absolut pengelompokan, dipilih dari tabel sensitivitas.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Uji cepat: split train dipotong, berkas ditulis ke folder sementara.",
    )
    args = parser.parse_args()
    if args.correlation_threshold is not None and not 0.0 < args.correlation_threshold < 1.0:
        parser.error("--correlation-threshold harus di antara 0 dan 1.")
    if args.smoke:
        # Hasil smoke tidak boleh menimpa kelompok yang dibaca consumer.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / FEATURE_GROUPS_PATH.name
            run_grouping(args.correlation_threshold, True, output)
            if output.exists():
                print(output.read_text()[:600])
    else:
        run_grouping(args.correlation_threshold, False, FEATURE_GROUPS_PATH)


if __name__ == "__main__":
    main()
