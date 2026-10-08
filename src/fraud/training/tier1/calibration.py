"""Kalibrasi skor mentah kandidat terbaik menjadi peluang, sebelum ambang keputusan ditetapkan.

Kandidat terbaik dipilih lintas semua kandidat (tidak diasumsikan salah satu) dari hasil phase 1
dan phase 2 di MLflow. Skor yang dibandingkan adalah rata-rata confirm (pelatihan ulang beberapa
seed) kalau tersedia, karena skor pencarian memakai seed yang sama dengan proses pencarian dan
cenderung optimistis. Run phase 2 menggantikan phase 1 hanya kalau unggul lebih dari
`NOISE_MULTIPLIER` galat baku, dan selisih antar kandidat yang berada di dalam rentang itu
dilaporkan sebagai peringatan (tidak menolak memilih: dua model yang setara dalam noise bisa
saling menggantikan, dan promosi ke produksi tetap butuh persetujuan manusia).

Model yang dikalibrasi adalah model yang sudah dilatih murni di split train, tidak dilatih ulang.
Kalibratornya (isotonic regression) di-fit di split validasi, split yang sama yang sudah
dipakai untuk memilih kandidat ini, mengikuti urutan kerja yang sudah ditetapkan: validasi
untuk seleksi dan kalibrasi, uji dilaporkan sekali di akhir.

Dijalankan lewat:
    uv run python -m fraud.training.tier1.calibration [--smoke]
"""

import argparse
import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import mlflow
import mlflow.sklearn as mlflow_sklearn
import numpy as np
import pandas as pd
from mlflow.entities import Run
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

from fraud.training.tier1.candidates import (
    CANDIDATE_NAMES,
    STAGE_PHASE1,
    STAGE_PHASE2,
    load_training_data,
    parent_run,
    setup_mlflow,
)
from fraud.training.tier1.dataset import Split
from fraud.training.tier1.search_results import (
    Confirmation,
    best_parent,
    confirmation_of,
    load_model,
    logged_model_uri,
)

STAGE_CALIBRATION = "calibration"

CALIBRATION_METHOD = "isotonic"
RELIABILITY_BINS = 10

# Selisih dua skor dianggap nyata hanya kalau lebih dari kelipatan ini dari galat bakunya.
NOISE_MULTIPLIER = 2.0

# Model terkalibrasi membungkus model asli di dalamnya, jadi daftar ini mencakup tipe dari
# keempat kandidat, diverifikasi langsung dengan mencoba menyimpan tiap jenis model yang
# sudah dikalibrasi (bukan ditebak dari pola nama kelas).
_TRUSTED_SKOPS_TYPES = [
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
    "sklearn.calibration._CalibratedClassifier",
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]
_log_calibrated_model = partial(mlflow_sklearn.log_model, skops_trusted_types=_TRUSTED_SKOPS_TYPES)


@dataclass(frozen=True)
class Contender:
    """Satu run sumber sebuah kandidat (phase 1 atau phase 2), dengan hasil confirm-nya."""

    name: str
    run: Run
    search_score: float
    confirmation: Confirmation | None

    @property
    def stage(self) -> str:
        return self.run.data.tags["search_stage"]


@dataclass(frozen=True)
class Comparison:
    """Hasil membandingkan dua kontestan.

    Attributes:
        leader: Kontestan dengan skor lebih tinggi.
        trailing: Kontestan lainnya.
        margin: Selisih skor keduanya, dalam basis yang sama.
        standard_error: Galat baku selisih dua rata-rata confirm, None kalau salah satu
            kontestan belum dikonfirmasi sehingga noise tidak bisa dinilai.
    """

    leader: Contender
    trailing: Contender
    margin: float
    standard_error: float | None

    @property
    def confirmed(self) -> bool:
        return self.standard_error is not None

    @property
    def within_noise(self) -> bool:
        if self.standard_error is None:
            return False
        return self.margin <= NOISE_MULTIPLIER * self.standard_error


@dataclass(frozen=True)
class BestCandidate:
    """Kandidat terpilih beserta perbandingannya dengan pesaing terdekat.

    Attributes:
        contender: Run sumber yang dikalibrasi.
        comparison: Perbandingan dengan pesaing terdekat dari kandidat lain, None kalau
            hanya ada satu kandidat yang punya hasil.
    """

    contender: Contender
    comparison: Comparison | None


def compare(a: Contender, b: Contender) -> Comparison:
    """Membandingkan dua kontestan, memakai rata-rata confirm kalau keduanya sudah dikonfirmasi.

    Kalau salah satu belum dikonfirmasi, keduanya dibandingkan lewat skor pencarian, dan galat
    baku tidak tersedia: mencampur rata-rata confirm dengan skor pencarian yang optimistis
    akan menguntungkan yang belum dikonfirmasi.
    """
    if a.confirmation and b.confirmation:
        score_a, score_b = a.confirmation.mean, b.confirmation.mean
        standard_error = math.sqrt(
            a.confirmation.std**2 / a.confirmation.n_seeds
            + b.confirmation.std**2 / b.confirmation.n_seeds
        )
    else:
        score_a, score_b, standard_error = a.search_score, b.search_score, None
    leader, trailing = (a, b) if score_a >= score_b else (b, a)
    return Comparison(leader, trailing, abs(score_a - score_b), standard_error)


def choose_source(incumbent: Contender, challenger: Contender) -> Contender:
    """Memilih antara run phase 1 (petahana) dan phase 2 (penantang) milik satu kandidat.

    Penantang menggantikan petahana hanya kalau unggul lebih dari `NOISE_MULTIPLIER` galat
    baku pada rata-rata confirm. Kalau belum keduanya dikonfirmasi, tidak ada dasar menilai
    noise, jadi petahana dipertahankan: run phase 2 sudah melewati lebih banyak seleksi pada
    seed yang sama, sehingga skornya lebih rentan optimistis.
    """
    comparison = compare(incumbent, challenger)
    if comparison.leader is challenger and comparison.confirmed and not comparison.within_noise:
        return challenger
    return incumbent


def _top(contenders: list[Contender]) -> Contender:
    """Kontestan teratas lewat perbandingan berpasangan."""
    top = contenders[0]
    for other in contenders[1:]:
        top = compare(top, other).leader
    return top


def _contenders(experiment_id: str, name: str) -> list[Contender]:
    """Run sumber terbaik tiap stage (phase 1 lalu phase 2) untuk satu kandidat."""
    contenders = []
    for stage in (STAGE_PHASE1, STAGE_PHASE2):
        try:
            run = best_parent(experiment_id, name, (stage,))
        except RuntimeError:
            continue
        confirmation = confirmation_of(experiment_id, name, run.info.run_id)
        contenders.append(
            Contender(name, run, run.data.metrics["best_validation_pr_auc"], confirmation)
        )
    return contenders


def find_best_candidate(experiment_id: str) -> BestCandidate:
    """Mencari kandidat terbaik lintas semua kandidat, dengan memperhitungkan noise.

    Args:
        experiment_id: Eksperimen MLflow.

    Returns:
        Kandidat terbaik lintas seluruh `CANDIDATE_NAMES` yang punya hasil, tanpa
        mengasumsikan satu kandidat tertentu yang menang, beserta perbandingannya dengan
        pesaing terdekat.

    Raises:
        RuntimeError: Kalau belum ada satu pun kandidat dengan hasil phase 1 atau phase 2.
    """
    winners = []
    for name in CANDIDATE_NAMES:
        contenders = _contenders(experiment_id, name)
        if not contenders:
            continue
        winner = contenders[0]
        for challenger in contenders[1:]:
            winner = choose_source(winner, challenger)
            if winner is not challenger:
                print(
                    f"  {name}: {challenger.stage} {challenger.run.info.run_id[:8]} tidak "
                    f"menggantikan {winner.stage} {winner.run.info.run_id[:8]}",
                    flush=True,
                )
        winners.append(winner)
    if not winners:
        raise RuntimeError("Belum ada kandidat dengan hasil phase 1 atau phase 2 yang selesai.")

    leader = _top(winners)
    rest = [winner for winner in winners if winner is not leader]
    comparison = compare(leader, _top(rest)) if rest else None
    return BestCandidate(leader, comparison)


def print_selection(best: BestCandidate) -> None:
    """Mencetak kandidat terpilih, dasar pemilihannya, dan peringatan kalau ada."""
    contender = best.contender
    confirmation = contender.confirmation
    if confirmation:
        basis = (
            f"rata-rata confirm {confirmation.mean:.4f} +/- {confirmation.std:.4f} "
            f"({confirmation.n_seeds} seed)"
        )
    else:
        basis = f"skor pencarian {contender.search_score:.4f} (belum dikonfirmasi)"
    print(
        f"Kandidat terbaik: {contender.name} ({contender.stage} "
        f"{contender.run.info.run_id[:8]}), {basis}",
        flush=True,
    )
    if not confirmation:
        print("PERINGATAN: belum ada confirm, skor pencarian cenderung optimistis.", flush=True)

    comparison = best.comparison
    if comparison is None:
        return
    other = comparison.trailing if comparison.leader is contender else comparison.leader
    line = (
        f"Pesaing terdekat: {other.name} ({other.stage} {other.run.info.run_id[:8]}), "
        f"selisih {comparison.margin:.4f}"
    )
    if comparison.standard_error is None:
        print(f"{line}, noise tidak bisa dinilai karena belum keduanya dikonfirmasi.", flush=True)
        print("PERINGATAN: pemenang belum terbukti melampaui noise.", flush=True)
        return
    multiple = comparison.margin / comparison.standard_error if comparison.standard_error else 0.0
    print(f"{line}, galat baku {comparison.standard_error:.4f} ({multiple:.1f} kali).", flush=True)
    if comparison.within_noise:
        print(
            f"PERINGATAN: selisih di dalam {NOISE_MULTIPLIER:g} galat baku, kedua kandidat "
            "setara dalam noise dan pemenang bukan klaim kemenangan.",
            flush=True,
        )


def selection_metrics(best: BestCandidate) -> dict[str, float]:
    """Metrik dasar pemilihan kandidat untuk dicatat di MLflow."""
    contender = best.contender
    metrics = {"source_validation_pr_auc": contender.search_score}
    if contender.confirmation:
        metrics["selection_confirm_mean"] = contender.confirmation.mean
        metrics["selection_confirm_std"] = contender.confirmation.std
    if best.comparison:
        metrics["margin_to_runner_up"] = best.comparison.margin
        if best.comparison.standard_error is not None:
            metrics["standard_error_to_runner_up"] = best.comparison.standard_error
    return metrics


def winner_within_noise_tag(best: BestCandidate) -> str:
    """Nilai tag `winner_within_noise` untuk sebuah pemilihan kandidat.

    Returns:
        "true" atau "false" kalau noise bisa dinilai, "unknown" kalau belum keduanya
        dikonfirmasi, "not_applicable" kalau hanya ada satu kandidat.
    """
    if best.comparison is None:
        return "not_applicable"
    if not best.comparison.confirmed:
        return "unknown"
    return str(best.comparison.within_noise).lower()


def latest_calibration_run(experiment_id: str) -> Run:
    """Run kalibrasi terbaru yang sudah selesai, apa pun kandidat asalnya.

    Berbeda dengan pemilihan kandidat, kalibrasi tidak punya skor untuk diperingkat: yang
    berlaku selalu hasil kalibrasi terakhir.

    Raises:
        RuntimeError: Kalau belum ada run kalibrasi yang selesai.
    """
    runs = mlflow.MlflowClient().search_runs(
        [experiment_id],
        filter_string=(
            f"tags.search_stage = '{STAGE_CALIBRATION}' and tags.run_role = 'parent' "
            "and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("Belum ada run kalibrasi yang selesai di eksperimen ini.")
    return runs[0]


def load_calibrated_model(run: Run) -> CalibratedClassifierCV:
    """Memuat model terkalibrasi dari sebuah run kalibrasi.

    Selalu memakai loader sklearn karena model terkalibrasi berformat sklearn, apa pun
    kandidat yang dibungkusnya.
    """
    model = mlflow_sklearn.load_model(logged_model_uri(run))
    assert isinstance(model, CalibratedClassifierCV)
    return model


def reliability_table(
    label: pd.Series, probability: np.ndarray, n_bins: int = RELIABILITY_BINS
) -> pd.DataFrame:
    """Rata-rata peluang prediksi vs fraud rate aktual per kelompok skor, urut menaik.

    Sanity check kalibrasi: kalau kalibrasi bekerja, kedua kolom saling mendekati di tiap
    baris. Kelompok dibagi rata jumlah barisnya, bukan rata lebar skornya, supaya tiap
    kelompok punya cukup baris untuk fraud rate aktualnya bermakna secara statistik.

    Args:
        label: Label fraud sebenarnya (0 atau 1).
        probability: Peluang prediksi, sejajar dengan `label`.
        n_bins: Jumlah kelompok.

    Returns:
        Satu baris per kelompok: jumlah baris, rata-rata peluang prediksi, dan fraud rate
        aktual di kelompok itu.
    """
    order = np.argsort(probability)
    rows = [
        {
            "n_rows": len(bin_indices),
            "mean_predicted": float(probability[bin_indices].mean()),
            "actual_fraud_rate": float(label.iloc[bin_indices].mean()),
        }
        for bin_indices in np.array_split(order, n_bins)
    ]
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class CalibrationResult:
    """Model terkalibrasi beserta ukuran mutu kalibrasinya di split validasi.

    Attributes:
        model: Model asli yang dibungkus kalibrator isotonic.
        brier_before: Brier score skor mentah.
        brier_after: Brier score skor terkalibrasi.
        reliability: Tabel rata-rata prediksi vs fraud rate aktual per kelompok skor.
    """

    model: CalibratedClassifierCV
    brier_before: float
    brier_after: float
    reliability: pd.DataFrame


def fit_isotonic(model: Any, validation: Split) -> CalibrationResult:
    """Mengalibrasi model yang sudah dilatih dengan isotonic regression di split validasi.

    Model asli dibekukan sehingga tidak dilatih ulang; hanya kalibratornya yang di-fit.

    Args:
        model: Model terlatih dengan `predict_proba`.
        validation: Split validasi, tempat kalibrator di-fit dan mutunya diukur.
    """
    raw_scores = model.predict_proba(validation.features)[:, 1]
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=CALIBRATION_METHOD)
    calibrated.fit(validation.features, validation.label)
    calibrated_scores = calibrated.predict_proba(validation.features)[:, 1]
    return CalibrationResult(
        model=calibrated,
        brier_before=float(brier_score_loss(validation.label, raw_scores)),
        brier_after=float(brier_score_loss(validation.label, calibrated_scores)),
        reliability=reliability_table(validation.label, calibrated_scores),
    )


def print_calibration(result: CalibrationResult) -> None:
    """Mencetak Brier score sebelum dan sesudah kalibrasi beserta tabel reliability."""
    print(
        f"Brier score: sebelum {result.brier_before:.5f}, sesudah {result.brier_after:.5f}",
        flush=True,
    )
    print(result.reliability.to_string(index=False), flush=True)


def log_calibration(result: CalibrationResult) -> None:
    """Mencatat mutu kalibrasi dan model terkalibrasi di run MLflow yang sedang aktif."""
    mlflow.log_metrics({"brier_before": result.brier_before, "brier_after": result.brier_after})
    mlflow.log_dict(
        {"rows": result.reliability.to_dict(orient="records")}, "reliability_table.json"
    )
    _log_calibrated_model(result.model, name="model")


def run_calibration(experiment_id: str, smoke: bool) -> CalibratedClassifierCV:
    """Mengalibrasi kandidat terbaik dan mencatatnya sebagai run baru di MLflow.

    Args:
        experiment_id: Eksperimen MLflow tempat hasil phase 1 dan phase 2 dibaca.
        smoke: True untuk memotong data, hanya untuk uji coba cepat.

    Returns:
        Model terkalibrasi. Brier score sebelum dan sesudah kalibrasi, tabel reliability,
        dan model itu sendiri tercatat di MLflow.
    """
    best = find_best_candidate(experiment_id)
    contender = best.contender
    print_selection(best)

    model = load_model(contender.run, contender.name)
    train, validation, spec = load_training_data(smoke)
    result = fit_isotonic(model, validation)
    print_calibration(result)

    with parent_run(contender.name, STAGE_CALIBRATION, train, validation, spec):
        mlflow.log_params(
            {
                "source_run_id": contender.run.info.run_id,
                "source_stage": contender.stage,
                "source_candidate": contender.name,
                "method": CALIBRATION_METHOD,
                "selection_basis": "confirm_mean" if contender.confirmation else "search_score",
            }
        )
        mlflow.log_metrics(selection_metrics(best))
        mlflow.set_tag("winner_within_noise", winner_within_noise_tag(best))
        log_calibration(result)
    return result.model


def main() -> None:
    parser = argparse.ArgumentParser(description="Kalibrasi kandidat terbaik tier 1.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Uji cepat: data dipotong dan dicatat di eksperimen terpisah.",
    )
    args = parser.parse_args()

    experiment_id = setup_mlflow(args.smoke)
    run_calibration(experiment_id, args.smoke)


if __name__ == "__main__":
    main()
