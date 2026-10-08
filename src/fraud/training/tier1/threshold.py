"""Penetapan ambang keputusan tier 1 dari skor terkalibrasi, dengan analisis sensitivitas biaya.

Ambang tolak diturunkan dari rasio biaya, ambang review dari kapasitas review. Kedua asumsi itu
tidak tersedia dari data, jadi dievaluasi lewat beberapa skenario dan dilaporkan sebagai tabel
sensitivitas, bukan diklaim sebagai satu angka benar. Semua perhitungan memakai split validasi,
split uji tidak pernah dibaca di sini.

Aturan keputusan untuk skor terkalibrasi s:
    s >= ambang tolak                  -> tolak otomatis
    ambang review <= s < ambang tolak  -> setujui dan tandai untuk review
    s < ambang review                  -> setujui biasa

Ambang bawah (batas pemanggilan model tier 2) sengaja tidak dihitung: butuh eksperimen dengan
model tier 2 yang belum dibangun, dan mengisinya dengan angka sementara berisiko dianggap final.

Dijalankan lewat:
    uv run python -m fraud.training.tier1.threshold [--cost-ratio N] [--smoke] [--output PATH]
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import numpy as np

from fraud.training.tier1.calibration import latest_calibration_run, load_calibrated_model
from fraud.training.tier1.candidates import load_training_data, parent_run, setup_mlflow

STAGE_THRESHOLD = "threshold"

# Rasio biaya lolos-fraud terhadap biaya salah-tolak, dipilih supaya ambang jatuh di skor nyata.
COST_RATIOS = (2, 5, 10, 20)
DEFAULT_COST_RATIO = 5

# Laju review yang sanggup ditangani analis, sebagai fraksi volume transaksi.
REVIEW_CAPACITY_FRACTION = 0.01

DEFAULT_OUTPUT_PATH = Path("configs/tier1_thresholds.json")

DECISION_RULE = (
    "tolak jika skor >= reject; setujui dan tandai review jika review <= skor < reject; "
    "selain itu setujui biasa"
)

LOWER_THRESHOLD_NOTE = (
    "belum ditentukan: memerlukan eksperimen dengan model tier 2 yang belum dibangun"
)

ASSUMPTIONS = (
    "Rasio biaya lolos-fraud terhadap biaya salah-tolak tidak tersedia dari data, jadi "
    "dievaluasi sebagai skenario, bukan diklaim sebagai satu angka benar.",
    "Kapasitas review adalah laju (fraksi volume transaksi yang sanggup direview), bukan batas "
    "keras: review boleh selesai di hari berikutnya, tetapi rata-rata laju antrean masuk tidak "
    "boleh melebihi laju review, kalau tidak backlog tumbuh tanpa batas.",
    "Transaksi di bawah ambang review tidak pernah masuk antrean review, dan itu risiko yang "
    "diterima secara sadar.",
    "Batas waktu penyelesaian review belum didefinisikan, jadi umur antrean tidak dievaluasi "
    "di sini dan perlu dipantau terpisah.",
    "Kalibrator dan ambang dipilih pada split validasi yang sama, jadi precision dan recall "
    "di tabel ini bisa meleset beberapa poin persentase dari kenyataan. Angka resmi diukur "
    "sekali di data uji tanpa mengubah ambang.",
)


@dataclass(frozen=True)
class ReviewChoice:
    """Hasil pemilihan ambang review untuk satu ambang tolak.

    Attributes:
        threshold: Ambang review terpilih.
        queue_rows: Jumlah transaksi yang masuk antrean review dengan ambang itu.
        next_plateau_rows: Ukuran blok skor kembar berikutnya yang tidak muat di kapasitas,
            gambaran seberapa kasar granularitas antrean.
        zone_empty: True kalau blok skor kembar tertinggi di bawah ambang tolak sendiri sudah
            melebihi kapasitas, sehingga tidak ada transaksi yang masuk antrean.
        covers_all_below_reject: True kalau kapasitas cukup untuk seluruh transaksi di bawah
            ambang tolak, sehingga tidak ada lagi transaksi yang disetujui biasa.
    """

    threshold: float
    queue_rows: int
    next_plateau_rows: int
    zone_empty: bool
    covers_all_below_reject: bool


@dataclass(frozen=True)
class ScenarioResult:
    """Hasil evaluasi satu skenario rasio biaya pada data validasi.

    Attributes:
        cost_ratio: Biaya lolos-fraud dibanding biaya salah-tolak.
        reject_threshold: Ambang tolak dari rumus biaya.
        review_threshold: Ambang review dari kapasitas.
        reject_rate: Fraksi transaksi yang ditolak.
        reject_precision: Fraksi fraud di antara yang ditolak, None kalau tidak ada yang ditolak.
        reject_recall: Fraksi seluruh fraud yang tertolak.
        review_target_rate: Kapasitas review yang dituju, fraksi dari seluruh transaksi.
        review_actual_rate: Ukuran antrean review sebenarnya, fraksi dari seluruh transaksi.
        next_plateau_rate: Ukuran blok skor kembar berikutnya, fraksi dari seluruh transaksi.
        review_precision: Fraksi fraud di antrean review, None kalau antreannya kosong.
        review_recall: Fraksi seluruh fraud yang masuk antrean review.
        unreviewed_fraud_share: Fraksi seluruh fraud yang lolos tanpa ditolak maupun direview.
        review_zone_empty: Lihat `ReviewChoice.zone_empty`.
        review_covers_all_below_reject: Lihat `ReviewChoice.covers_all_below_reject`.
    """

    cost_ratio: float
    reject_threshold: float
    review_threshold: float
    reject_rate: float
    reject_precision: float | None
    reject_recall: float
    review_target_rate: float
    review_actual_rate: float
    next_plateau_rate: float
    review_precision: float | None
    review_recall: float
    unreviewed_fraud_share: float
    review_zone_empty: bool
    review_covers_all_below_reject: bool


def reject_threshold_for(cost_ratio: float) -> float:
    """Ambang tolak dari rasio biaya lolos-fraud terhadap biaya salah-tolak.

    Setara dengan biaya salah-tolak / (biaya salah-tolak + biaya lolos-fraud), ditulis dengan
    rasio karena hanya perbandingan kedua biaya itu yang bisa diasumsikan.
    """
    return 1.0 / (1.0 + cost_ratio)


def choose_review_threshold(
    score: np.ndarray, reject_threshold: float, capacity_fraction: float
) -> ReviewChoice:
    """Memilih ambang review supaya antrean tidak melebihi kapasitas yang dituju.

    Kalibrasi isotonic menghasilkan fungsi tangga: banyak transaksi berbagi skor yang persis
    sama, jadi antrean hanya bisa bertambah per blok skor kembar, bukan per transaksi. Blok
    ditambahkan dari skor tertinggi di bawah ambang tolak ke bawah, selama total antrean tidak
    melebihi kapasitas (aturan "paling banyak", karena laju antrean masuk tidak boleh melebihi
    laju review). Ambang ditaruh di tengah antara blok terendah yang masuk dan blok tertinggi
    yang tidak masuk, supaya perbandingan tetap benar walau skor dibulatkan ke float32 saat
    model diekspor.

    Args:
        score: Skor terkalibrasi seluruh transaksi validasi.
        reject_threshold: Ambang tolak, batas atas zona review.
        capacity_fraction: Kapasitas review sebagai fraksi dari seluruh transaksi.

    Returns:
        Ambang review terpilih beserta ukuran antrean sebenarnya.
    """
    below_reject = score[score < reject_threshold]
    values, counts = np.unique(below_reject, return_counts=True)
    values, counts = values[::-1], counts[::-1]
    cumulative = np.cumsum(counts)
    included = int(np.searchsorted(cumulative, capacity_fraction * len(score), side="right"))

    if included == 0:
        next_rows = int(counts[0]) if len(counts) else 0
        return ReviewChoice(reject_threshold, 0, next_rows, True, False)
    if included == len(values):
        return ReviewChoice(float(values[-1]), int(cumulative[-1]), 0, False, True)
    midpoint = float((values[included - 1] + values[included]) / 2)
    return ReviewChoice(
        midpoint, int(cumulative[included - 1]), int(counts[included]), False, False
    )


def _fraud_rate(is_fraud: np.ndarray, mask: np.ndarray) -> float | None:
    """Fraksi fraud di antara baris yang dipilih `mask`, None kalau tidak ada baris."""
    return float(is_fraud[mask].mean()) if mask.any() else None


def _fraud_share(is_fraud: np.ndarray, mask: np.ndarray) -> float:
    """Fraksi seluruh fraud yang berada di baris yang dipilih `mask`."""
    return float(is_fraud[mask].sum() / max(int(is_fraud.sum()), 1))


def evaluate_scenario(
    score: np.ndarray, label: np.ndarray, cost_ratio: float, capacity_fraction: float
) -> ScenarioResult:
    """Menghitung ambang dan dampaknya pada data validasi untuk satu rasio biaya.

    Args:
        score: Skor terkalibrasi seluruh transaksi validasi.
        label: Label fraud sebenarnya (0 atau 1), sejajar dengan `score`.
        cost_ratio: Biaya lolos-fraud dibanding biaya salah-tolak.
        capacity_fraction: Kapasitas review sebagai fraksi dari seluruh transaksi.

    Returns:
        Ambang tolak dan review, plus reject rate, precision, recall, dan ukuran antrean.
    """
    reject_threshold = reject_threshold_for(cost_ratio)
    review = choose_review_threshold(score, reject_threshold, capacity_fraction)
    is_fraud = label == 1
    rejected = score >= reject_threshold
    in_review = (score >= review.threshold) & ~rejected
    unreviewed = score < review.threshold
    total = len(score)

    return ScenarioResult(
        cost_ratio=cost_ratio,
        reject_threshold=reject_threshold,
        review_threshold=review.threshold,
        reject_rate=float(rejected.mean()),
        reject_precision=_fraud_rate(is_fraud, rejected),
        reject_recall=_fraud_share(is_fraud, rejected),
        review_target_rate=capacity_fraction,
        review_actual_rate=review.queue_rows / total,
        next_plateau_rate=review.next_plateau_rows / total,
        review_precision=_fraud_rate(is_fraud, in_review),
        review_recall=_fraud_share(is_fraud, in_review),
        unreviewed_fraud_share=_fraud_share(is_fraud, unreviewed),
        review_zone_empty=review.zone_empty,
        review_covers_all_below_reject=review.covers_all_below_reject,
    )


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def print_sensitivity_table(results: list[ScenarioResult], distinct_scores: int) -> None:
    """Mencetak tabel sensitivitas semua skenario, dengan catatan asumsi di bawahnya."""
    print(f"Skor terkalibrasi bernilai unik: {distinct_scores} (fungsi tangga isotonic)")
    print(
        f"{'rasio':>5} {'tolak>=':>8} {'review>=':>9} {'reject%':>8} {'prec':>7} {'recall':>7} "
        f"{'antrean%':>9} {'target%':>8} {'blok+1%':>8} {'rev.prec':>8} {'lolos%':>7}"
    )
    for r in results:
        print(
            f"{r.cost_ratio:>4}:1 {r.reject_threshold:>8.4f} {r.review_threshold:>9.4f} "
            f"{_percent(r.reject_rate):>8} {_percent(r.reject_precision):>7} "
            f"{_percent(r.reject_recall):>7} {_percent(r.review_actual_rate):>9} "
            f"{_percent(r.review_target_rate):>8} {_percent(r.next_plateau_rate):>8} "
            f"{_percent(r.review_precision):>8} {_percent(r.unreviewed_fraud_share):>7}"
        )
    print("Asumsi:")
    for assumption in ASSUMPTIONS:
        print(f"  - {assumption}")


def build_config(
    results: list[ScenarioResult],
    selected: ScenarioResult,
    capacity_fraction: float,
    candidate: str,
    source_run_id: str,
) -> dict:
    """Menyusun isi file konfigurasi ambang yang dibaca lapisan serving.

    Args:
        results: Hasil semua skenario.
        selected: Skenario yang dipakai sebagai titik operasi.
        capacity_fraction: Kapasitas review yang diasumsikan.
        candidate: Nama kandidat yang skornya dikalibrasi.
        source_run_id: Run kalibrasi asal skor.

    Returns:
        Dict siap di-serialisasi ke JSON: titik operasi terpilih, ambang tiap skenario, dan
        asumsi yang melatarbelakanginya.
    """
    return {
        "candidate": candidate,
        "source_calibration_run_id": source_run_id,
        "selected_cost_ratio": selected.cost_ratio,
        "decision_rule": DECISION_RULE,
        "thresholds": {
            "reject": selected.reject_threshold,
            "review": selected.review_threshold,
            "lower": None,
        },
        "lower_threshold_note": LOWER_THRESHOLD_NOTE,
        "review_capacity_fraction": capacity_fraction,
        "scenarios": {
            str(r.cost_ratio): {
                "reject": r.reject_threshold,
                "review": r.review_threshold,
                "reject_rate": r.reject_rate,
                "review_actual_rate": r.review_actual_rate,
                "usable": not (r.review_zone_empty or r.review_covers_all_below_reject),
            }
            for r in results
        },
        "assumptions": list(ASSUMPTIONS),
    }


def compute_thresholds(
    score: np.ndarray, label: np.ndarray, cost_ratio: int, capacity_fraction: float
) -> tuple[list[ScenarioResult], ScenarioResult]:
    """Menghitung ambang semua skenario biaya dan memilih skenario titik operasi.

    Args:
        score: Skor terkalibrasi di split validasi.
        label: Label fraud sejajar dengan `score`.
        cost_ratio: Skenario yang dipakai sebagai titik operasi, salah satu `COST_RATIOS`.
        capacity_fraction: Kapasitas review sebagai fraksi dari seluruh transaksi.

    Returns:
        Hasil semua skenario dan skenario terpilih.
    """
    results = [
        evaluate_scenario(score, label, ratio, capacity_fraction) for ratio in COST_RATIOS
    ]
    return results, next(r for r in results if r.cost_ratio == cost_ratio)


def ensure_meaningful(selected: ScenarioResult) -> None:
    """Menolak skenario yang membuat tiga kelas keputusan tidak bermakna.

    Raises:
        ValueError: Kalau zona review kosong atau menghabiskan seluruh transaksi di bawah
            ambang tolak.
    """
    if selected.review_zone_empty or selected.review_covers_all_below_reject:
        raise ValueError(
            f"Skenario {selected.cost_ratio}:1 menghasilkan zona review yang tidak bermakna "
            f"(kosong={selected.review_zone_empty}, "
            f"menghabiskan semua={selected.review_covers_all_below_reject})."
        )


def threshold_metrics(selected: ScenarioResult, distinct_scores: int) -> dict[str, float]:
    """Metrik skenario terpilih untuk dicatat di MLflow."""
    metrics = {
        "reject_rate": selected.reject_rate,
        "reject_recall": selected.reject_recall,
        "review_actual_rate": selected.review_actual_rate,
        "review_recall": selected.review_recall,
        "unreviewed_fraud_share": selected.unreviewed_fraud_share,
        "distinct_scores": float(distinct_scores),
    }
    if selected.reject_precision is not None:
        metrics["reject_precision"] = selected.reject_precision
    if selected.review_precision is not None:
        metrics["review_precision"] = selected.review_precision
    return metrics


def run_threshold(
    experiment_id: str,
    smoke: bool,
    cost_ratio: int,
    output_path: Path,
    capacity_fraction: float = REVIEW_CAPACITY_FRACTION,
) -> ScenarioResult:
    """Menghitung ambang untuk semua skenario, menyimpan yang terpilih, dan mencatat auditnya.

    Args:
        experiment_id: Eksperimen MLflow tempat run kalibrasi dibaca.
        smoke: True untuk memotong data, hanya untuk uji coba cepat.
        cost_ratio: Skenario yang dipakai sebagai titik operasi, salah satu `COST_RATIOS`.
        output_path: Lokasi file konfigurasi yang ditulis.
        capacity_fraction: Kapasitas review sebagai fraksi dari seluruh transaksi.

    Returns:
        Hasil skenario terpilih.

    Raises:
        ValueError: Kalau skenario terpilih menghasilkan zona review kosong atau menghabiskan
            seluruh transaksi di bawah ambang tolak, yang membuat tiga kelas keputusan tidak
            bermakna. Tidak ada file yang ditulis dalam kasus ini.
    """
    calibration_run = latest_calibration_run(experiment_id)
    candidate = calibration_run.data.tags["candidate"]
    print(f"Kalibrasi sumber: {candidate} (run {calibration_run.info.run_id[:8]})", flush=True)

    model = load_calibrated_model(calibration_run)
    train, validation, spec = load_training_data(smoke)
    score = model.predict_proba(validation.features)[:, 1]
    label = validation.label.to_numpy()

    distinct_scores = len(np.unique(score))
    results, selected = compute_thresholds(score, label, cost_ratio, capacity_fraction)
    print_sensitivity_table(results, distinct_scores)
    ensure_meaningful(selected)

    config = build_config(
        results, selected, capacity_fraction, candidate, calibration_run.info.run_id
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(f"Konfigurasi ditulis ke {output_path} (skenario {cost_ratio}:1)", flush=True)

    metrics = threshold_metrics(selected, distinct_scores)

    with parent_run(candidate, STAGE_THRESHOLD, train, validation, spec):
        mlflow.log_params(
            {
                "source_calibration_run_id": calibration_run.info.run_id,
                "source_candidate": candidate,
                "selected_cost_ratio": cost_ratio,
                "review_capacity_fraction": capacity_fraction,
                "cost_ratios": ",".join(str(ratio) for ratio in COST_RATIOS),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_dict(
            {
                "assumptions": list(ASSUMPTIONS),
                "scenarios": [asdict(r) for r in results],
            },
            "sensitivity_table.json",
        )
        mlflow.log_artifact(str(output_path))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Tetapkan ambang keputusan tier 1.")
    parser.add_argument(
        "--cost-ratio",
        type=int,
        choices=COST_RATIOS,
        default=DEFAULT_COST_RATIO,
        help="Skenario rasio biaya (lolos-fraud : salah-tolak) yang dipakai sebagai titik operasi.",
    )
    parser.add_argument(
        "--review-capacity",
        type=float,
        default=REVIEW_CAPACITY_FRACTION,
        help="Kapasitas review sebagai fraksi volume transaksi (0 sampai 1, bukan persen).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Lokasi file konfigurasi ambang yang ditulis.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Uji cepat: data dipotong dan dicatat di eksperimen terpisah.",
    )
    args = parser.parse_args()
    if args.smoke and args.output == DEFAULT_OUTPUT_PATH:
        parser.error("--smoke tidak boleh menimpa file konfigurasi produksi, beri --output lain.")

    if not 0 < args.review_capacity < 1:
        parser.error("--review-capacity harus berupa fraksi di antara 0 dan 1, misalnya 0.01.")

    experiment_id = setup_mlflow(args.smoke)
    run_threshold(
        experiment_id, args.smoke, args.cost_ratio, args.output, args.review_capacity
    )


if __name__ == "__main__":
    main()
