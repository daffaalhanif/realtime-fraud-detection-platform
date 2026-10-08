"""Evaluasi akhir model tier 1 terdaftar: PR-AUC per segmen dan kinerja pada ambang yang dibekukan.

Dijalankan sekali pada split uji (20% terakhir secara waktu) untuk satu versi model. Pemakaian
split uji tidak boleh diulang: hasilnya dipakai untuk mengubah sesuatu lalu diukur lagi, angkanya
tidak lagi mewakili data yang belum pernah dilihat. Karena itu run resmi ditolak kalau sudah ada
run evaluasi uji yang selesai untuk versi model yang sama, dan tidak ada flag untuk melewatinya.

Model yang dievaluasi adalah ONNX dari registry (artefak yang akan melayani), dan ambangnya
dibaca dari berkas yang tersimpan bersama versi model itu, tidak disetel ulang di sini.

Laporan dibagi dua kelompok segmen:
    ukuran entitas     total transaksi per kunci entitas sepanjang dataset (kecil, sedang, besar)
    posisi histori     urutan transaksi dalam histori entitasnya (1, 2-5, 6-20, 21 atau lebih)

Dijalankan lewat:
    uv run python -m fraud.training.tier1.evaluate --model-version N [--smoke]

Dengan `--smoke`, kode yang sama berjalan pada split validasi dan dicatat di eksperimen uji,
sehingga bisa dicoba berulang tanpa menyentuh split uji.
"""

import argparse
import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import mlflow.artifacts as mlflow_artifacts
import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud.training.tier1.candidates import RANDOM_SEED, setup_mlflow
from fraud.training.tier1.dataset import PreparedData, Split, prepare_datasets
from fraud.training.tier1.export_onnx import REGISTERED_MODEL_NAME

# Telemetri bawaan onnxruntime membuat proses abort saat penutupan dan memanggil jaringan
ort.disable_telemetry_events()

STAGE_TEST_EVALUATION = "test_evaluation"

BOOTSTRAP_RESAMPLES = 1000
CONFIDENCE_LEVEL = 0.95

# Batas ukuran entitas: kecil dan besar mengikuti eksplorasi data awal, sedang adalah sisanya
SIZE_SMALL_MAX = 5
SIZE_MEDIUM_MAX = 50

# Batas atas posisi histori tiga kelompok pertama, sisanya masuk "21 atau lebih"
POSITION_FIRST = 1
POSITION_SHORT_MAX = 5
POSITION_MEDIUM_MAX = 20

SIZE_SEGMENTS = (
    ("size_small", "Entitas kecil (5 transaksi atau kurang)"),
    ("size_medium", "Entitas sedang (6-50 transaksi)"),
    ("size_large", "Entitas besar (lebih dari 50 transaksi)"),
)
POSITION_SEGMENTS = (
    ("position_1", "Posisi histori 1"),
    ("position_2_5", "Posisi histori 2-5"),
    ("position_6_20", "Posisi histori 6-20"),
    ("position_21_plus", "Posisi histori 21 atau lebih"),
)

ASSUMPTIONS = (
    "Ukuran entitas adalah total transaksi per kunci entitas sepanjang dataset (train, validasi, "
    "dan uji). Ukuran ini memakai transaksi yang belum terjadi saat suatu transaksi dinilai, jadi "
    "hanya dipakai untuk mengelompokkan laporan dan tidak pernah menjadi fitur model.",
    "Batas kecil (5 transaksi atau kurang) dan besar (lebih dari 50) mengikuti eksplorasi data "
    "awal. Kelompok sedang (6-50) adalah asumsi eksplisit, karena hanya dua batas itu yang "
    "pernah didefinisikan.",
    "Posisi histori dihitung dari seluruh dataset (jumlah transaksi sebelumnya + 1). Posisi 1 "
    "berarti transaksi pertama kunci entitas itu dalam jendela data, bukan pemilik yang baru.",
    f"Selang kepercayaan {CONFIDENCE_LEVEL:.0%} dari bootstrap {BOOTSTRAP_RESAMPLES} resampel "
    f"per segmen (seed {RANDOM_SEED}). Resampel tanpa satu pun fraud dilewati, dan jumlah "
    "resampel yang valid dilaporkan. Segmen dengan sedikit fraud tetap punya selang lebar.",
    "Ambang dibekukan dari versi model dan tidak disetel ulang. Angka validasi pembanding "
    "cenderung optimistis karena ambang dipilih pada data itu.",
)


@dataclass(frozen=True)
class SegmentMetrics:
    """Kinerja model pada satu segmen.

    Attributes:
        key: Kunci segmen yang stabil, dipakai sebagai nama metrik.
        label: Nama segmen untuk dibaca manusia.
        n_rows: Jumlah transaksi di segmen.
        n_fraud: Jumlah transaksi fraud di segmen.
        fraud_rate: Fraksi fraud, None kalau segmen kosong.
        pr_auc: PR-AUC (average precision), None kalau segmen tidak punya fraud.
        ci_low: Batas bawah selang kepercayaan bootstrap, None kalau tidak bisa dihitung.
        ci_high: Batas atas selang kepercayaan bootstrap, None kalau tidak bisa dihitung.
        n_valid_resamples: Resampel bootstrap yang memuat minimal satu fraud.
    """

    key: str
    label: str
    n_rows: int
    n_fraud: int
    fraud_rate: float | None
    pr_auc: float | None
    ci_low: float | None
    ci_high: float | None
    n_valid_resamples: int


@dataclass(frozen=True)
class ThresholdReport:
    """Kinerja satu skenario rasio biaya pada ambang beku.

    Attributes:
        cost_ratio: Rasio biaya lolos-fraud terhadap salah-tolak, sebagai teks kunci skenario.
        reject_threshold: Ambang tolak dari versi model.
        review_threshold: Ambang review dari versi model.
        usable: Penanda dari konfigurasi ambang: zona review tidak kosong dan tidak menelan semua.
        reject_rate: Fraksi transaksi yang ditolak.
        validation_reject_rate: Reject rate yang tercatat di konfigurasi (dari validasi).
        reject_precision: Fraksi fraud di antara yang ditolak, None kalau tidak ada yang ditolak.
        reject_recall: Fraksi seluruh fraud yang tertolak, None kalau tidak ada fraud.
        review_rate: Ukuran antrean review sebagai fraksi dari seluruh transaksi.
        validation_review_rate: Ukuran antrean yang tercatat di konfigurasi (dari validasi).
        review_target_rate: Kapasitas review yang dituju di konfigurasi.
        review_precision: Fraksi fraud di antrean review, None kalau antreannya kosong.
        review_recall: Fraksi seluruh fraud yang masuk antrean, None kalau tidak ada fraud.
        unreviewed_fraud_share: Fraksi seluruh fraud yang lolos tanpa ditolak maupun direview,
            None kalau tidak ada fraud.
    """

    cost_ratio: str
    reject_threshold: float
    review_threshold: float
    usable: bool
    reject_rate: float
    validation_reject_rate: float
    reject_precision: float | None
    reject_recall: float | None
    review_rate: float
    validation_review_rate: float
    review_target_rate: float
    review_precision: float | None
    review_recall: float | None
    unreviewed_fraud_share: float | None


@dataclass(frozen=True)
class EvaluationReport:
    """Laporan evaluasi lengkap satu versi model pada satu split."""

    model_name: str
    model_version: int
    candidate: str
    source_calibration_run_id: str
    evaluated_split: str
    n_rows: int
    n_fraud: int
    fraud_rate: float
    overall: SegmentMetrics
    entity_size: list[SegmentMetrics]
    history_position: list[SegmentMetrics]
    thresholds: list[ThresholdReport]
    assumptions: list[str]


@dataclass(frozen=True)
class RegisteredModel:
    """Model terdaftar yang siap dievaluasi beserta berkas yang menyertainya."""

    session: ort.InferenceSession
    version: int
    candidate: str
    source_calibration_run_id: str
    thresholds: dict
    feature_spec: dict


def _bootstrap_interval(
    score: np.ndarray, is_fraud: np.ndarray
) -> tuple[float | None, float | None, int]:
    """Selang kepercayaan PR-AUC dengan bootstrap baris di dalam satu segmen.

    Args:
        score: Skor model segmen, harus memuat minimal satu fraud.
        is_fraud: Label fraud boolean sejajar dengan `score`.

    Returns:
        Batas bawah, batas atas (keduanya None kalau tidak ada resampel valid), dan jumlah
        resampel valid. Resampel tanpa fraud dilewati karena PR-AUC-nya tidak terdefinisi.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_rows = len(score)
    values = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        chosen = rng.integers(0, n_rows, n_rows)
        if is_fraud[chosen].any():
            values.append(average_precision_score(is_fraud[chosen], score[chosen]))
    if not values:
        return None, None, 0
    tail = (1 - CONFIDENCE_LEVEL) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high), len(values)


def segment_metrics(
    key: str, label: str, score: np.ndarray, is_fraud: np.ndarray
) -> SegmentMetrics:
    """Menghitung PR-AUC dan selang bootstrap satu segmen, dengan penanganan segmen tanpa fraud.

    Args:
        key: Kunci segmen yang stabil.
        label: Nama segmen untuk dibaca manusia.
        score: Skor terkalibrasi transaksi di segmen.
        is_fraud: Label fraud boolean sejajar dengan `score`.

    Returns:
        Metrik segmen. Segmen kosong atau tanpa fraud dilaporkan tanpa PR-AUC, bukan dengan nol.
    """
    n_rows = len(score)
    n_fraud = int(is_fraud.sum())
    fraud_rate = n_fraud / n_rows if n_rows else None
    if n_fraud == 0:
        return SegmentMetrics(key, label, n_rows, 0, fraud_rate, None, None, None, 0)
    low, high, valid = _bootstrap_interval(score, is_fraud)
    pr_auc = float(average_precision_score(is_fraud, score))
    return SegmentMetrics(key, label, n_rows, n_fraud, fraud_rate, pr_auc, low, high, valid)


def size_segment_keys(entity_totals: np.ndarray) -> np.ndarray:
    """Memetakan total transaksi per entitas ke kunci segmen ukuran."""
    return np.select(
        [entity_totals <= SIZE_SMALL_MAX, entity_totals <= SIZE_MEDIUM_MAX],
        ["size_small", "size_medium"],
        default="size_large",
    )


def position_segment_keys(txn_count_so_far: np.ndarray) -> np.ndarray:
    """Memetakan jumlah transaksi sebelumnya ke kunci segmen posisi histori."""
    position = txn_count_so_far + 1
    return np.select(
        [
            position == POSITION_FIRST,
            position <= POSITION_SHORT_MAX,
            position <= POSITION_MEDIUM_MAX,
        ],
        ["position_1", "position_2_5", "position_6_20"],
        default="position_21_plus",
    )


def evaluate_segments(
    score: np.ndarray,
    is_fraud: np.ndarray,
    keys: np.ndarray,
    definitions: tuple[tuple[str, str], ...],
) -> list[SegmentMetrics]:
    """Menghitung metrik tiap segmen berurutan sesuai `definitions`, termasuk yang kosong."""
    return [
        segment_metrics(key, label, score[keys == key], is_fraud[keys == key])
        for key, label in definitions
    ]


def _fraud_fraction(is_fraud: np.ndarray, mask: np.ndarray) -> float | None:
    """Fraksi fraud di antara baris yang dipilih `mask`, None kalau tidak ada baris."""
    return float(is_fraud[mask].mean()) if mask.any() else None


def _share_of_fraud(is_fraud: np.ndarray, mask: np.ndarray) -> float | None:
    """Fraksi seluruh fraud yang berada di baris `mask`, None kalau tidak ada fraud sama sekali."""
    total = int(is_fraud.sum())
    return float(is_fraud[mask].sum() / total) if total else None


def threshold_report(
    score: np.ndarray, is_fraud: np.ndarray, cost_ratio: str, scenario: dict, capacity: float
) -> ThresholdReport:
    """Mengukur satu skenario pada ambang beku dengan aturan keputusan tiga kelas.

    Args:
        score: Skor terkalibrasi seluruh transaksi yang dievaluasi.
        is_fraud: Label fraud boolean sejajar dengan `score`.
        cost_ratio: Kunci skenario di konfigurasi ambang.
        scenario: Isi skenario itu di konfigurasi: ambang, dan angka validasi pembanding.
        capacity: Kapasitas review yang dituju, fraksi dari seluruh transaksi.

    Returns:
        Reject rate, ukuran antrean, precision, dan recall. Nilai yang penyebutnya nol
        dilaporkan None supaya tidak terbaca sebagai kinerja nol.
    """
    reject, review = scenario["reject"], scenario["review"]
    rejected = score >= reject
    in_review = (score >= review) & ~rejected
    unreviewed = score < review
    return ThresholdReport(
        cost_ratio=cost_ratio,
        reject_threshold=reject,
        review_threshold=review,
        usable=scenario["usable"],
        reject_rate=float(rejected.mean()),
        validation_reject_rate=scenario["reject_rate"],
        reject_precision=_fraud_fraction(is_fraud, rejected),
        reject_recall=_share_of_fraud(is_fraud, rejected),
        review_rate=float(in_review.mean()),
        validation_review_rate=scenario["review_actual_rate"],
        review_target_rate=capacity,
        review_precision=_fraud_fraction(is_fraud, in_review),
        review_recall=_share_of_fraud(is_fraud, in_review),
        unreviewed_fraud_share=_share_of_fraud(is_fraud, unreviewed),
    )


def check_edge_cases() -> None:
    """Menguji cabang tepi pada data sintetis sebelum split evaluasi dibaca.

    Dijalankan di awal setiap run karena pemakaian split uji tidak bisa diulang: bug pada
    segmen kosong, segmen tanpa fraud, atau penyebut nol harus berhenti di sini, bukan setelah
    data uji terpakai.

    Raises:
        ValueError: Kalau salah satu pemeriksaan gagal, dengan daftar semua yang gagal.
    """
    problems: list[str] = []

    def expect(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    def flags(*values: int) -> np.ndarray:
        return np.array(values, dtype=bool)

    empty = segment_metrics("k", "k", np.array([]), flags())
    expect(empty.n_rows == 0 and empty.pr_auc is None, "segmen kosong harus tanpa PR-AUC")
    expect(empty.fraud_rate is None, "fraud rate segmen kosong harus None")

    no_fraud = segment_metrics("k", "k", np.array([0.1, 0.2]), flags(0, 0))
    expect(no_fraud.pr_auc is None and no_fraud.ci_low is None, "segmen tanpa fraud harus None")
    expect(no_fraud.fraud_rate == 0.0, "fraud rate segmen tanpa fraud harus 0")

    all_fraud = segment_metrics("k", "k", np.array([0.3, 0.9]), flags(1, 1))
    expect(all_fraud.pr_auc == 1.0, "segmen yang seluruhnya fraud harus PR-AUC 1")

    tiny = segment_metrics("k", "k", np.array([0.1, 0.2, 0.3, 0.4, 0.9]), flags(0, 0, 0, 0, 1))
    expect(
        tiny.pr_auc is not None and 0 < tiny.n_valid_resamples < BOOTSTRAP_RESAMPLES,
        "segmen dengan satu fraud harus melewati resampel tanpa fraud dan tetap punya selang",
    )
    expect(
        tiny.ci_low is not None and tiny.ci_high is not None and tiny.ci_low <= tiny.ci_high,
        "selang segmen kecil harus terurut",
    )

    tied = segment_metrics("k", "k", np.full(10, 0.5), flags(1, 1, 0, 0, 0, 0, 0, 0, 0, 0))
    expect(
        tied.pr_auc is not None and abs(tied.pr_auc - 0.2) < 1e-12,
        "skor kembar: PR-AUC harus sama dengan fraud rate",
    )

    expect(
        list(size_segment_keys(np.array([1, 5, 6, 50, 51])))
        == ["size_small", "size_small", "size_medium", "size_medium", "size_large"],
        "batas segmen ukuran salah",
    )
    expect(
        list(position_segment_keys(np.array([0, 1, 4, 5, 19, 20])))
        == ["position_1", "position_2_5", "position_2_5", "position_6_20", "position_6_20",
            "position_21_plus"],
        "batas segmen posisi salah",
    )

    scenario = {"reject": 0.8, "review": 0.5, "usable": True, "reject_rate": 0.0,
                "review_actual_rate": 0.0}
    score = np.array([0.9, 0.6, 0.2, 0.1])
    normal = threshold_report(score, flags(1, 0, 1, 0), "5", scenario, 0.01)
    expect(
        normal.reject_precision == 1.0 and normal.reject_recall == 0.5, "precision dan recall tolak"
    )
    expect(
        normal.review_precision == 0.0 and normal.unreviewed_fraud_share == 0.5,
        "precision review dan fraud yang lolos",
    )

    nobody_rejected = threshold_report(
        score, flags(1, 0, 1, 0), "5", {**scenario, "reject": 0.99, "review": 0.99}, 0.01
    )
    expect(
        nobody_rejected.reject_precision is None and nobody_rejected.reject_recall == 0.0,
        "tidak ada yang ditolak: precision None, recall 0",
    )
    expect(
        nobody_rejected.review_precision is None and nobody_rejected.review_rate == 0.0,
        "antrean review kosong: precision None",
    )

    without_fraud = threshold_report(score, flags(0, 0, 0, 0), "5", scenario, 0.01)
    expect(
        without_fraud.reject_recall is None and without_fraud.unreviewed_fraud_share is None,
        "tanpa fraud sama sekali: recall dan fraud lolos harus None",
    )

    if problems:
        raise ValueError("Pemeriksaan kasus tepi gagal: " + "; ".join(problems))


def load_registered_model(version: int, name: str = REGISTERED_MODEL_NAME) -> RegisteredModel:
    """Memuat satu versi model dari registry beserta ambang dan kontrak inputnya.

    Args:
        version: Nomor versi model terdaftar.
        name: Nama model di registry.

    Returns:
        Sesi ONNX, kandidat, run kalibrasi asal, serta isi `thresholds.json` dan
        `feature_spec.json` yang tersimpan bersama versi itu.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = Path(
            mlflow_artifacts.download_artifacts(
                f"models:/{name}/{version}", dst_path=directory
            )
        )
        thresholds = json.loads((path / "extra_files" / "thresholds.json").read_text())
        feature_spec = json.loads((path / "extra_files" / "feature_spec.json").read_text())
        session = ort.InferenceSession(
            (path / "model.onnx").read_bytes(), providers=["CPUExecutionProvider"]
        )
    return RegisteredModel(
        session=session,
        version=version,
        candidate=thresholds["candidate"],
        source_calibration_run_id=thresholds["source_calibration_run_id"],
        thresholds=thresholds,
        feature_spec=feature_spec,
    )


def completed_evaluations(experiment_id: str, version: int, split_name: str) -> list[str]:
    """Id run evaluasi selesai untuk satu versi model pada satu split (dasar penjaga)."""
    runs = mlflow.MlflowClient().search_runs(
        [experiment_id],
        filter_string=(
            f"tags.search_stage = '{STAGE_TEST_EVALUATION}' "
            f"and params.model_version = '{version}' "
            f"and params.evaluated_split = '{split_name}' "
            "and attributes.status = 'FINISHED'"
        ),
    )
    return [run.info.run_id for run in runs]


def entity_totals(data: PreparedData, split: Split) -> np.ndarray:
    """Total transaksi per kunci entitas di seluruh dataset, dipetakan ke baris `split`."""
    counts = pd.concat(
        [part.entity_key["card1"] for part in (data.train, data.validation, data.test)]
    ).value_counts()
    return split.entity_key["card1"].map(counts).to_numpy()


def predict_scores(model: RegisteredModel, features: pd.DataFrame) -> np.ndarray:
    """Skor terkalibrasi seluruh baris dari model ONNX, sebagai float64."""
    outputs = model.session.run(
        ["calibrated_probability"], {"input": features.to_numpy(dtype=np.float32)}
    )
    return np.asarray(outputs[0], dtype=np.float64)


def build_report(
    model: RegisteredModel, split_name: str, data: PreparedData, split: Split
) -> EvaluationReport:
    """Menyusun laporan lengkap: PR-AUC global dan per segmen, serta kinerja pada ambang beku."""
    score = predict_scores(model, split.features)
    is_fraud = split.label.to_numpy() == 1
    n_fraud = int(is_fraud.sum())

    print(f"Bootstrap {BOOTSTRAP_RESAMPLES} resampel per segmen...", flush=True)
    overall = segment_metrics("overall", "Global", score, is_fraud)
    by_size = evaluate_segments(
        score, is_fraud, size_segment_keys(entity_totals(data, split)), SIZE_SEGMENTS
    )
    positions = split.features["txn_count_so_far"].to_numpy().astype(np.int64)
    by_position = evaluate_segments(
        score, is_fraud, position_segment_keys(positions), POSITION_SEGMENTS
    )

    capacity = model.thresholds["review_capacity_fraction"]
    thresholds = [
        threshold_report(score, is_fraud, ratio, scenario, capacity)
        for ratio, scenario in model.thresholds["scenarios"].items()
    ]
    return EvaluationReport(
        model_name=REGISTERED_MODEL_NAME,
        model_version=model.version,
        candidate=model.candidate,
        source_calibration_run_id=model.source_calibration_run_id,
        evaluated_split=split_name,
        n_rows=len(score),
        n_fraud=n_fraud,
        fraud_rate=n_fraud / len(score),
        overall=overall,
        entity_size=by_size,
        history_position=by_position,
        thresholds=thresholds,
        assumptions=list(ASSUMPTIONS),
    )


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def print_report(report: EvaluationReport) -> None:
    """Mencetak tabel PR-AUC per segmen, tabel ambang beku, dan asumsinya."""
    print(
        f"\n=== PR-AUC, split {report.evaluated_split}: {report.n_rows} baris, "
        f"fraud {report.fraud_rate:.2%} ==="
    )
    print(
        f"{'segmen':<42} {'baris':>8} {'fraud':>6} {'fraud%':>7} {'PR-AUC':>7} "
        f"{'selang 95%':>16}"
    )
    for segment in [report.overall, *report.entity_size, *report.history_position]:
        interval = (
            "-"
            if segment.ci_low is None or segment.ci_high is None
            else f"{segment.ci_low:.4f}-{segment.ci_high:.4f}"
        )
        print(
            f"{segment.label:<42} {segment.n_rows:>8} {segment.n_fraud:>6} "
            f"{_percent(segment.fraud_rate):>7} {_number(segment.pr_auc):>7} {interval:>16}"
        )

    print(f"\n=== Ambang beku, split {report.evaluated_split} (dalam kurung: validasi) ===")
    print(
        f"{'rasio':>5} {'reject%':>16} {'prec':>7} {'recall':>7} {'antrean%':>16} "
        f"{'target%':>8} {'rev.prec':>8} {'rev.rec':>8} {'lolos%':>7}"
    )
    for t in report.thresholds:
        reject_rate = f"{_percent(t.reject_rate)} ({_percent(t.validation_reject_rate)})"
        review_rate = f"{_percent(t.review_rate)} ({_percent(t.validation_review_rate)})"
        print(
            f"{t.cost_ratio:>4}:1 {reject_rate:>16} {_percent(t.reject_precision):>7} "
            f"{_percent(t.reject_recall):>7} {review_rate:>16} {_percent(t.review_target_rate):>8} "
            f"{_percent(t.review_precision):>8} {_percent(t.review_recall):>8} "
            f"{_percent(t.unreviewed_fraud_share):>7}"
        )
    print("Asumsi:")
    for assumption in report.assumptions:
        print(f"  - {assumption}")


def _flat_metrics(report: EvaluationReport) -> dict[str, float]:
    """Meratakan laporan jadi metrik MLflow, hanya nilai yang terdefinisi."""
    metrics: dict[str, float] = {}
    segments = [report.overall, *report.entity_size, *report.history_position]
    for s in segments:
        metrics[f"{s.key}_n_rows"] = float(s.n_rows)
        metrics[f"{s.key}_n_fraud"] = float(s.n_fraud)
        for suffix, value in (("pr_auc", s.pr_auc), ("pr_auc_ci_low", s.ci_low),
                              ("pr_auc_ci_high", s.ci_high)):
            if value is not None:
                metrics[f"{s.key}_{suffix}"] = value
    for t in report.thresholds:
        values = {
            "reject_rate": t.reject_rate,
            "reject_precision": t.reject_precision,
            "reject_recall": t.reject_recall,
            "review_rate": t.review_rate,
            "review_precision": t.review_precision,
            "review_recall": t.review_recall,
            "unreviewed_fraud_share": t.unreviewed_fraud_share,
        }
        metrics.update({f"ratio{t.cost_ratio}_{k}": v for k, v in values.items() if v is not None})
    return metrics


def run_evaluation(experiment_id: str, model_version: int, smoke: bool) -> EvaluationReport:
    """Mengevaluasi satu versi model terdaftar, mencetak laporan, dan mencatatnya ke MLflow.

    Args:
        experiment_id: Eksperimen MLflow tempat run evaluasi dicatat.
        model_version: Versi model terdaftar yang dievaluasi.
        smoke: True untuk mengevaluasi split validasi, bukan uji, dan tidak menegakkan penjaga
            sekali pakai.

    Returns:
        Laporan evaluasi.

    Raises:
        ValueError: Kalau pemeriksaan kasus tepi gagal, atau kontrak input model tidak sama
            dengan pemetaan data saat ini. Split evaluasi belum dibaca dalam kasus ini.
        RuntimeError: Kalau evaluasi uji untuk versi ini sudah pernah selesai (bukan smoke).
    """
    split_name = "validation" if smoke else "test"
    check_edge_cases()
    print("Pemeriksaan kasus tepi lulus.", flush=True)

    model = load_registered_model(model_version)
    print(
        f"Model {REGISTERED_MODEL_NAME} versi {model_version}: {model.candidate}, "
        f"kalibrasi {model.source_calibration_run_id[:8]}",
        flush=True,
    )

    previous = completed_evaluations(experiment_id, model_version, split_name)
    if previous and not smoke:
        raise RuntimeError(
            f"Evaluasi uji versi {model_version} sudah selesai (run {previous[0]}). Split uji "
            "hanya boleh dipakai sekali. Kalau memang harus diulang, hapus run itu di MLflow "
            "secara sadar."
        )
    if previous:
        print(f"Run evaluasi {split_name} sebelumnya: {len(previous)} (smoke tidak dibatasi).")

    start = time.perf_counter()
    data = prepare_datasets()
    if json.loads(json.dumps(dict(data.spec))) != model.feature_spec:
        raise ValueError("Kontrak input model berbeda dari pemetaan data saat ini.")
    split = data.validation if smoke else data.test
    print(f"Data siap: {time.perf_counter() - start:.0f} detik, split {split_name}.", flush=True)

    start = time.perf_counter()
    report = build_report(model, split_name, data, split)
    evaluation_seconds = time.perf_counter() - start
    print_report(report)
    print(f"\nWaktu prediksi dan bootstrap: {evaluation_seconds:.0f} detik.")

    with mlflow.start_run(run_name=f"{model.candidate}-{STAGE_TEST_EVALUATION}"):
        mlflow.set_tags(
            {
                "candidate": model.candidate,
                "search_stage": STAGE_TEST_EVALUATION,
                "run_role": "parent",
            }
        )
        mlflow.log_params(
            {
                "model_name": REGISTERED_MODEL_NAME,
                "model_version": model_version,
                "evaluated_split": split_name,
                "source_calibration_run_id": model.source_calibration_run_id,
                "n_rows": report.n_rows,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "random_seed": RANDOM_SEED,
            }
        )
        mlflow.log_metrics({**_flat_metrics(report), "evaluation_seconds": evaluation_seconds})
        mlflow.log_dict(asdict(report), f"{split_name}_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluasi akhir model tier 1 terdaftar.")
    parser.add_argument(
        "--model-version",
        type=int,
        required=True,
        help="Versi model terdaftar yang dievaluasi (eksplisit, bukan alias).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Uji: evaluasi split validasi, dicatat di eksperimen terpisah, dapat diulang.",
    )
    args = parser.parse_args()

    experiment_id = setup_mlflow(args.smoke)
    run_evaluation(experiment_id, args.model_version, args.smoke)


if __name__ == "__main__":
    main()
