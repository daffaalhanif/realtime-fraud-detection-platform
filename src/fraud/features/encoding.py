"""Penyusunan input model dari data transaksi, dipakai jalur training maupun serving.

Pemetaan nilai kategorikal ke kode dilatih sekali dari data training, lalu disimpan bersama
model sebagai artefak. Pelatihan dan scoring real-time sama-sama menyusun matriks input lewat
`build_model_input`, supaya urutan kolom dan kode kategorikal tidak pernah berbeda antara
saat model dilatih dan saat dipakai (training-serving skew).
"""

from collections.abc import Mapping, Sequence
from typing import TypedDict

import numpy as np
import pandas as pd

CategoryMappings = dict[str, dict[str, int]]


class FeatureSpec(TypedDict):
    """Kontrak input model, disimpan sebagai artefak bersama model.

    Attributes:
        input_columns: Urutan kolom yang diterima model, fitur agregat lebih dulu.
        encoded_columns: Kolom di `input_columns` yang semula string, kini kode ordinal.
        nominal_numeric_columns: Kolom di `input_columns` yang nominal tapi tetap bernilai
            asli. Model pohon boleh memakainya apa adanya, model linear perlu one-hot.
        category_mappings: Pemetaan nilai ke kode untuk tiap kolom di `encoded_columns`.
    """

    input_columns: list[str]
    encoded_columns: list[str]
    nominal_numeric_columns: list[str]
    category_mappings: CategoryMappings


def fit_category_mappings(df: pd.DataFrame, columns: list[str]) -> CategoryMappings:
    """Membangun pemetaan nilai ke kode ordinal untuk tiap kolom kategorikal.

    Hanya boleh dipanggil dengan data training. Nilai yang baru muncul di data
    validasi, uji, atau produksi sengaja tidak masuk pemetaan, supaya pemetaan tidak
    memuat informasi dari masa depan.

    Args:
        df: Data training yang memuat seluruh kolom di `columns`.
        columns: Nama kolom kategorikal yang dipetakan.

    Returns:
        Pemetaan per kolom, nilai kategori ke kode bulat mulai dari 0. Nilai kosong
        tidak diberi kode.
    """
    return {
        # Diurutkan supaya kode tidak bergantung urutan baris saat pelatihan ulang.
        column: {
            value: code
            for code, value in enumerate(sorted(df[column].dropna().unique()))
        }
        for column in columns
    }


def build_model_input(
    columns: Mapping[str, Sequence | np.ndarray], spec: FeatureSpec
) -> np.ndarray:
    """Menyusun matriks input model dari nilai mentah per kolom.

    Bekerja kolom per kolom supaya fungsi yang sama cukup cepat untuk satu transaksi saat
    scoring real-time maupun ratusan ribu baris saat pelatihan.

    Args:
        columns: Nilai per kolom untuk seluruh nama di `spec["input_columns"]`, semua
            sepanjang jumlah baris. Nilai kosong boleh `None` maupun NaN.
        spec: Kontrak input model.

    Returns:
        Matriks float32 berbentuk `(jumlah baris, jumlah kolom input)` dengan urutan kolom
        `spec["input_columns"]`. Kolom kategorikal berisi kode ordinal, NaN untuk nilai
        kosong maupun nilai yang tidak dikenal pemetaan.
    """
    names = spec["input_columns"]
    mappings = spec["category_mappings"]
    n_rows = len(columns[names[0]])
    matrix = np.empty((n_rows, len(names)), dtype=np.float32)
    for position, name in enumerate(names):
        values = columns[name]
        mapping = mappings.get(name)
        if mapping is None:
            matrix[:, position] = np.asarray(values, dtype=np.float64)
        else:
            matrix[:, position] = [mapping.get(value, np.nan) for value in values]
    return matrix
