"""Penyiapan data model tier 2: sequence transaksi mentah per kunci entitas, tanpa label di elemen.

Satu sampel adalah satu transaksi yang dinilai, didahului paling banyak `seq_len - 1` transaksi
sebelumnya pada kunci entitas yang sama. Label hanya milik transaksi yang dinilai, sehingga
elemen sequence sama dengan yang tersedia di Redis saat serving. Kolom mentah, pembagian split,
dan urutan kronologis memakai definisi tier 1 yang sama, supaya tier 2 dan pembandingnya
melihat informasi yang setara.

Seluruh transaksi di-encode sekali lewat `fraud.features.sequence`, lalu batch disusun dengan
operasi indeks vektor di device yang sama dengan model.
"""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from fraud.features.offline_store import INITIAL_PARQUET_DIR, LABEL_COLUMN
from fraud.features.online_store import SEQUENCE_RETENTION_LIMIT
from fraud.features.sequence import (
    MISSING_CODE,
    SequenceSpec,
    encode_elapsed,
    encode_transactions,
    fit_sequence_spec,
)
from fraud.training.tier1.dataset import (
    NUMERIC_CATEGORICAL_COLUMNS,
    load_transactions,
    raw_input_columns,
    split_temporal,
    string_columns,
)

SEQUENCE_MODES = ("ordered", "shuffled", "current_only")

# Sequence lebih panjang dari retensi Redis LIST tidak bisa disusun ulang saat serving.
MAX_SEQ_LEN = SEQUENCE_RETENTION_LIMIT

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SequenceData:
    """Seluruh transaksi yang sudah di-encode, urut per kunci entitas lalu kronologis.

    Baris sebelum sebuah baris pada urutan ini adalah histori kunci entitasnya, selama tidak
    melewati `entity_start` baris itu.

    Attributes:
        numeric: Nilai numerik ternormalisasi, float32 `(N, kolom numerik)`.
        missing: Indikator nilai numerik kosong, uint8 `(N, kolom numerik)`.
        categorical: Kode kategori, int64 `(N, kolom kategorikal)`.
        transaction_dt: `TransactionDT` tiap baris, int64 `(N,)`.
        label: Label fraud tiap baris, float32 `(N,)`. Hanya dipakai sebagai target sampel.
        entity_start: Posisi baris pertama kunci entitas tiap baris, int64 `(N,)`.
        split_rows: Posisi baris yang menjadi transaksi dinilai per split, menurut split
            temporal tier 1 atas urutan kronologis.
        transaction_id: `TransactionID` tiap baris, untuk evaluasi.
        card1: Kunci entitas tiap baris, untuk evaluasi per segmen.
        spec: Kontrak elemen sequence yang dipakai encoding.
    """

    numeric: torch.Tensor
    missing: torch.Tensor
    categorical: torch.Tensor
    transaction_dt: torch.Tensor
    label: torch.Tensor
    entity_start: torch.Tensor
    split_rows: dict[str, torch.Tensor]
    transaction_id: np.ndarray
    card1: np.ndarray
    spec: SequenceSpec

    def to(self, device: torch.device | str) -> "SequenceData":
        """Salinan dengan seluruh tensor dipindah ke `device`."""
        return replace(
            self,
            numeric=self.numeric.to(device),
            missing=self.missing.to(device),
            categorical=self.categorical.to(device),
            transaction_dt=self.transaction_dt.to(device),
            label=self.label.to(device),
            entity_start=self.entity_start.to(device),
            split_rows={name: rows.to(device) for name, rows in self.split_rows.items()},
        )


@dataclass(frozen=True)
class SequenceBatch:
    """Satu batch sampel, elemen terakhir tiap sequence adalah transaksi yang dinilai.

    Attributes:
        numeric: float32 `(B, L, kolom numerik)`.
        missing: uint8 `(B, L, kolom numerik)`.
        categorical: int64 `(B, L, kolom kategorikal)`.
        elapsed: Jarak waktu ter-encode tiap elemen ke transaksi yang dinilai, float32 `(B, L)`.
        padding_mask: True untuk posisi padding di kiri, bool `(B, L)`.
        label: Label fraud transaksi yang dinilai, float32 `(B,)`.
    """

    numeric: torch.Tensor
    missing: torch.Tensor
    categorical: torch.Tensor
    elapsed: torch.Tensor
    padding_mask: torch.Tensor
    label: torch.Tensor


def sequence_columns() -> tuple[list[str], list[str]]:
    """Kolom kategorikal dan numerik elemen sequence, dari kolom mentah input tier 1."""
    categorical = string_columns() + list(NUMERIC_CATEGORICAL_COLUMNS)
    numeric = [name for name in raw_input_columns() if name not in categorical]
    return categorical, numeric


def prepare_sequence_data(
    parquet_dir: Path = INITIAL_PARQUET_DIR, row_limit: int | None = None
) -> SequenceData:
    """Membaca Parquet, melatih kontrak elemen dari split train, dan meng-encode semua transaksi.

    Args:
        parquet_dir: Folder Parquet partisi per hari hasil pemuatan data awal.
        row_limit: Kalau diisi, hanya transaksi kronologis paling awal sebanyak ini yang
            dipakai, untuk uji coba cepat. Split dihitung ulang di atas potongan itu.

    Returns:
        Data siap disusun menjadi batch, masih di CPU.
    """
    df = load_transactions(parquet_dir)
    if row_limit is not None:
        df = df.iloc[:row_limit]
    train_part, validation_part, _ = split_temporal(df)
    train_end = len(train_part)
    validation_end = train_end + len(validation_part)

    categorical, numeric = sequence_columns()
    spec = fit_sequence_spec(train_part, categorical, numeric)
    encoded = encode_transactions(
        {name: df[name].to_numpy() for name in categorical + numeric}, spec
    )

    # Sort stabil mempertahankan urutan kronologis di dalam tiap kunci entitas.
    order = np.argsort(df["card1"].to_numpy(), kind="stable")
    card1 = df["card1"].to_numpy()[order]
    n_rows = len(order)
    is_first = np.ones(n_rows, dtype=bool)
    is_first[1:] = card1[1:] != card1[:-1]
    entity_start = np.maximum.accumulate(np.where(is_first, np.arange(n_rows), 0))

    split_rows = {
        "train": np.flatnonzero(order < train_end),
        "validation": np.flatnonzero((order >= train_end) & (order < validation_end)),
        "test": np.flatnonzero(order >= validation_end),
    }
    return SequenceData(
        numeric=torch.from_numpy(encoded.numeric[order]),
        missing=torch.from_numpy(encoded.missing[order]),
        categorical=torch.from_numpy(encoded.categorical[order]),
        transaction_dt=torch.from_numpy(df["TransactionDT"].to_numpy(dtype=np.int64)[order]),
        label=torch.from_numpy(df[LABEL_COLUMN].to_numpy(dtype=np.float32)[order]),
        entity_start=torch.from_numpy(entity_start),
        split_rows={name: torch.from_numpy(rows) for name, rows in split_rows.items()},
        transaction_id=df["TransactionID"].to_numpy()[order],
        card1=card1,
        spec=spec,
    )


def _shuffle_history(
    index: torch.Tensor,
    elapsed_seconds: torch.Tensor,
    padding: torch.Tensor,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mengacak elemen histori dan jarak waktunya dengan dua permutasi yang saling lepas.

    Elemen terakhir (transaksi yang dinilai) tidak ikut diacak. Jarak waktu diacak terpisah dari
    transaksinya, supaya urutan tidak bisa direkonstruksi dari waktu yang menempel di elemen.
    """
    history_padding = padding[:, :-1]

    def permutation() -> torch.Tensor:
        # Kunci acak dari generator CPU supaya hasilnya sama di CPU, MPS, maupun CUDA.
        keys = torch.rand(history_padding.shape, generator=generator).to(index.device)
        # Padding diberi kunci terkecil supaya tetap di kiri setelah diurutkan.
        return keys.masked_fill(history_padding, -1.0).argsort(dim=1)

    shuffled_index = index[:, :-1].gather(1, permutation())
    shuffled_elapsed = elapsed_seconds[:, :-1].gather(1, permutation())
    return (
        torch.cat([shuffled_index, index[:, -1:]], dim=1),
        torch.cat([shuffled_elapsed, elapsed_seconds[:, -1:]], dim=1),
    )


def build_batch(
    data: SequenceData,
    rows: torch.Tensor,
    seq_len: int,
    mode: str,
    generator: torch.Generator | None = None,
) -> SequenceBatch:
    """Menyusun satu batch sequence untuk transaksi-transaksi yang dinilai.

    Args:
        data: Data ter-encode, di device yang sama dengan `rows`.
        rows: Posisi baris transaksi yang dinilai, int64 `(B,)`.
        seq_len: Panjang sequence termasuk transaksi yang dinilai, 1 sampai `MAX_SEQ_LEN`.
        mode: Salah satu `SEQUENCE_MODES`. `current_only` selalu memakai panjang 1.
        generator: Generator CPU untuk mode `shuffled`; None memakai generator global.

    Returns:
        Batch berpadding kiri; posisi padding berisi nol dan ditandai `padding_mask`.

    Raises:
        ValueError: `mode` tidak dikenal atau `seq_len` di luar rentang.
    """
    if mode not in SEQUENCE_MODES:
        raise ValueError(f"Mode {mode!r} tidak dikenal, pilih salah satu {SEQUENCE_MODES}.")
    if not 1 <= seq_len <= MAX_SEQ_LEN:
        raise ValueError(f"seq_len {seq_len} di luar rentang 1 sampai {MAX_SEQ_LEN}.")
    length = 1 if mode == "current_only" else seq_len

    offsets = torch.arange(length - 1, -1, -1, device=rows.device)
    index = rows[:, None] - offsets[None, :]
    padding = index < data.entity_start[rows][:, None]
    # Posisi padding diarahkan ke baris yang valid supaya indeks aman, lalu isinya dinolkan.
    index = torch.where(padding, rows[:, None], index)
    elapsed_seconds = data.transaction_dt[rows][:, None] - data.transaction_dt[index]
    if mode == "shuffled" and length > 1:
        index, elapsed_seconds = _shuffle_history(index, elapsed_seconds, padding, generator)

    # Rumus jarak waktu dihitung lewat fungsi yang sama dengan serving, bukan ditulis ulang.
    elapsed = torch.from_numpy(encode_elapsed(elapsed_seconds.cpu().numpy(), data.spec))
    feature_padding = padding[:, :, None]
    return SequenceBatch(
        numeric=data.numeric[index].masked_fill(feature_padding, 0.0),
        missing=data.missing[index].masked_fill(feature_padding, 0),
        categorical=data.categorical[index].masked_fill(feature_padding, MISSING_CODE),
        elapsed=elapsed.to(rows.device).masked_fill(padding, 0.0),
        padding_mask=padding,
        label=data.label[rows],
    )


def iterate_rows(
    rows: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> Iterator[torch.Tensor]:
    """Membagi posisi baris menjadi batch, opsional diacak dengan generator CPU."""
    if shuffle:
        rows = rows[torch.randperm(len(rows), generator=generator).to(rows.device)]
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]
