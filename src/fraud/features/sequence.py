"""Encoding transaksi mentah menjadi elemen sequence model tier 2, dipakai training maupun serving.

Satu transaksi menjadi tiga bagian: nilai numerik yang sudah dinormalisasi, indikator nilai
kosong per kolom numerik, dan kode kategori per kolom kategorikal. Statistik normalisasi dan
pemetaan kategori dilatih sekali dari data training, lalu disimpan bersama model sebagai
`SequenceSpec`. Pelatihan dan scoring real-time sama-sama memakai `encode_transactions`, supaya
elemen sequence tidak pernah berbeda antara saat model dilatih dan saat dipakai.

Modul ini hanya bergantung pada numpy dan pandas, sehingga serving tidak membutuhkan PyTorch.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
import pandas as pd

MISSING_CODE = 0
RARE_CODE = 1
FIRST_CATEGORY_CODE = 2

# Kategori yang lebih jarang dari ini di data training digabung ke RARE_CODE, karena embedding
# untuk nilai yang hanya muncul beberapa kali tidak sempat terlatih dan rawan menghafal.
MIN_CATEGORY_COUNT = 100

# Pembagi log1p jarak waktu, supaya nilainya berada di sekitar 0 sampai 1 tanpa bergantung pada
# panjang sequence. Rentang dataset kurang dari setahun.
ELAPSED_LOG_SCALE = math.log1p(365 * 86_400)

# Kolom yang konstan di training bisa menghasilkan simpangan baku sekecil galat pembulatan float,
# bukan nol persis; pembagian dengannya membuat nilai di periode lain meledak.
CONSTANT_STD_TOLERANCE = 1e-6

# Nilai langka pada kolom yang hampir konstan menjadi sekitar 1/sqrt(peluangnya) setelah
# distandardisasi. Dibatasi supaya kejadian yang lebih jarang dari sekitar 1% tidak mendominasi
# proyeksi input, tanpa menghilangkan tanda maupun jaraknya dari nilai umum.
STANDARDIZED_CLIP = 10.0


class SequenceSpec(TypedDict):
    """Kontrak elemen sequence model tier 2, disimpan sebagai artefak bersama model.

    Attributes:
        categorical_columns: Kolom yang di-encode sebagai kode kategori, urut sesuai kolom output.
        numeric_columns: Kolom yang di-encode sebagai nilai numerik, urut sesuai kolom output.
        category_codes: Pemetaan kunci nilai ke kode per kolom kategorikal. Kunci dibentuk
            `category_key`, kode dimulai dari `FIRST_CATEGORY_CODE`.
        numeric_mean: Rata-rata nilai numerik setelah log bertanda, per kolom numerik.
        numeric_std: Simpangan baku nilai numerik setelah log bertanda, per kolom numerik.
        min_category_count: Batas kemunculan minimal kategori di data training.
        elapsed_log_scale: Pembagi log1p jarak waktu antar transaksi.
        standardized_clip: Batas mutlak nilai numerik setelah distandardisasi.
    """

    categorical_columns: list[str]
    numeric_columns: list[str]
    category_codes: dict[str, dict[str, int]]
    numeric_mean: list[float]
    numeric_std: list[float]
    min_category_count: int
    elapsed_log_scale: float
    standardized_clip: float


@dataclass(frozen=True)
class EncodedTransactions:
    """Elemen sequence hasil encoding sejumlah transaksi.

    Attributes:
        numeric: Nilai numerik ternormalisasi, float32 `(baris, kolom numerik)`, 0 untuk kosong.
        missing: Indikator nilai numerik kosong, uint8 berbentuk sama dengan `numeric`.
        categorical: Kode kategori, int64 `(baris, kolom kategorikal)`.
    """

    numeric: np.ndarray
    missing: np.ndarray
    categorical: np.ndarray


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def category_key(value: Any) -> str:
    """Kunci teks sebuah nilai kategori, sama untuk nilai dari pandas, Parquet, maupun JSON.

    Kode kategori yang tersimpan numerik bisa terbaca `111.0` dari Parquet tetapi `111` dari
    JSON, sehingga bilangan bulat diseragamkan tanpa desimal.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    # Log bertanda: meredam ekor panjang tanpa membuang tanda pada kolom yang bisa negatif.
    return np.sign(values) * np.log1p(np.abs(values))


def fit_sequence_spec(
    train: pd.DataFrame, categorical_columns: list[str], numeric_columns: list[str]
) -> SequenceSpec:
    """Membangun kontrak elemen sequence dari data training.

    Hanya boleh dipanggil dengan data training, supaya pemetaan dan statistik normalisasi tidak
    memuat informasi dari periode validasi, uji, maupun produksi.

    Args:
        train: Transaksi training yang memuat seluruh kolom di kedua daftar.
        categorical_columns: Kolom yang di-encode sebagai kode kategori.
        numeric_columns: Kolom yang di-encode sebagai nilai numerik.

    Returns:
        Kontrak elemen sequence siap disimpan sebagai JSON.
    """
    category_codes = {}
    for column in categorical_columns:
        counts = train[column].dropna().map(category_key).value_counts()
        # Diurutkan supaya kode tidak bergantung urutan baris saat pelatihan ulang.
        frequent = sorted(str(key) for key, count in counts.items() if count >= MIN_CATEGORY_COUNT)
        category_codes[column] = {
            key: code for code, key in enumerate(frequent, start=FIRST_CATEGORY_CODE)
        }

    transformed = _signed_log1p(train[numeric_columns].to_numpy(dtype=np.float64))
    mean = np.nanmean(transformed, axis=0)
    std = np.nanstd(transformed, axis=0)
    # Kolom yang seluruhnya kosong atau konstan di training tidak boleh menghasilkan NaN.
    mean = np.where(np.isnan(mean), 0.0, mean)
    std = np.where(np.isnan(std) | (std < CONSTANT_STD_TOLERANCE), 1.0, std)

    return SequenceSpec(
        categorical_columns=list(categorical_columns),
        numeric_columns=list(numeric_columns),
        category_codes=category_codes,
        numeric_mean=mean.tolist(),
        numeric_std=std.tolist(),
        min_category_count=MIN_CATEGORY_COUNT,
        elapsed_log_scale=ELAPSED_LOG_SCALE,
        standardized_clip=STANDARDIZED_CLIP,
    )


def encode_transactions(
    columns: Mapping[str, Sequence | np.ndarray], spec: SequenceSpec
) -> EncodedTransactions:
    """Meng-encode transaksi mentah menjadi elemen sequence.

    Bekerja kolom per kolom supaya fungsi yang sama cukup cepat untuk beberapa transaksi saat
    scoring real-time maupun seluruh dataset saat pelatihan.

    Args:
        columns: Nilai per kolom untuk seluruh kolom di spec, semua sepanjang jumlah baris.
            Nilai kosong boleh `None` maupun NaN.
        spec: Kontrak elemen sequence.

    Returns:
        Elemen sequence per transaksi. Kategori kosong menjadi `MISSING_CODE`, kategori yang
        tidak dikenal pemetaan menjadi `RARE_CODE`.
    """
    numeric_names = spec["numeric_columns"]
    raw = np.column_stack(
        [np.asarray(columns[name], dtype=np.float64) for name in numeric_names]
    )
    missing = np.isnan(raw)
    standardized = (_signed_log1p(raw) - np.asarray(spec["numeric_mean"])) / np.asarray(
        spec["numeric_std"]
    )
    clip = spec["standardized_clip"]
    numeric = np.where(missing, 0.0, np.clip(standardized, -clip, clip)).astype(np.float32)

    categorical_names = spec["categorical_columns"]
    categorical = np.empty((len(raw), len(categorical_names)), dtype=np.int64)
    for position, name in enumerate(categorical_names):
        codes = spec["category_codes"][name]
        categorical[:, position] = [
            MISSING_CODE if _is_missing(value) else codes.get(category_key(value), RARE_CODE)
            for value in columns[name]
        ]

    return EncodedTransactions(
        numeric=numeric, missing=missing.astype(np.uint8), categorical=categorical
    )


def encode_elapsed(elapsed_seconds: np.ndarray, spec: SequenceSpec) -> np.ndarray:
    """Meng-encode jarak waktu dari tiap elemen ke transaksi yang sedang dinilai.

    Args:
        elapsed_seconds: Selisih `TransactionDT` transaksi yang dinilai dikurangi transaksi
            elemen, bentuk apa pun. Nol untuk transaksi yang dinilai itu sendiri.
        spec: Kontrak elemen sequence.

    Returns:
        Array float32 berbentuk sama, `log1p(detik) / elapsed_log_scale`.
    """
    clipped = np.maximum(np.asarray(elapsed_seconds, dtype=np.float64), 0.0)
    return (np.log1p(clipped) / spec["elapsed_log_scale"]).astype(np.float32)
