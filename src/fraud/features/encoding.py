"""Encoding kolom kategorikal menjadi kode ordinal, dipakai jalur training maupun serving.

Pemetaan nilai ke kode dilatih sekali dari data training, lalu disimpan bersama model
sebagai artefak. Scoring real-time menerapkan pemetaan yang persis sama lewat fungsi
di modul ini, supaya nilai kategorikal yang sama tidak pernah menghasilkan kode berbeda
antara saat model dilatih dan saat dipakai (training-serving skew).
"""

import pandas as pd

CategoryMappings = dict[str, dict[str, int]]


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


def apply_category_mappings(df: pd.DataFrame, mappings: CategoryMappings) -> pd.DataFrame:
    """Mengganti nilai kategorikal dengan kode ordinal hasil `fit_category_mappings`.

    Args:
        df: Data yang kolom-kolomnya ada di `mappings`. Kolom lain tidak diubah.
        mappings: Pemetaan hasil `fit_category_mappings`.

    Returns:
        DataFrame baru. Kolom kategorikal berisi kode bertipe float32, NaN untuk nilai
        kosong maupun nilai yang tidak dikenal pemetaan. `df` asli tidak berubah.
    """
    return df.assign(
        **{
            column: df[column].map(mapping).astype("float32")
            for column, mapping in mappings.items()
        }
    )
