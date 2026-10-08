"""Pipeline tier 1 dengan fitur jendela waktu (T1+) menjadi versi baru `fraud-tier1` di registry.

Model sumbernya adalah pengulangan T1+ dengan PR-AUC validasi tertinggi dari pembanding adil.
Langkahnya sama persis dengan versi produksi pertama dan memakai fungsi yang sama: kalibrasi
isotonic di split validasi, ambang keputusan dari skenario rasio biaya, lalu ekspor ONNX mandiri
dengan gerbang konsistensi. Versi ini didaftarkan tanpa alias; memindahkan alias produksi tetap
keputusan manusia. Split uji tidak dibaca.

Semua run dicatat di eksperimen pengujian tier 2, tempat T1+ dilatih, bukan di eksperimen tuning
tier 1, supaya langkah tier 1 yang mengambil kalibrasi terbaru di sana tidak tertukar sumbernya.

Dijalankan lewat:
    uv run python -m fraud.training.tier1.window_variant [--cost-ratio N] [--smoke]
"""

import argparse
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import mlflow
import mlflow.lightgbm as mlflow_lightgbm
import numpy as np
import onnxruntime as ort
from lightgbm import LGBMClassifier
from mlflow.entities import Run

from fraud.training.tier1.calibration import (
    CALIBRATION_METHOD,
    STAGE_CALIBRATION,
    fit_isotonic,
    log_calibration,
    print_calibration,
)
from fraud.training.tier1.candidates import parent_run
from fraud.training.tier1.dataset import Split, prepare_datasets
from fraud.training.tier1.export_onnx import (
    ONNX_OPSET,
    REGISTERED_MODEL_NAME,
    SMOKE_REGISTERED_MODEL_NAME,
    STAGE_EXPORT,
    build_calibrated_onnx,
    check_consistency,
    export_metrics,
    log_and_register,
    measure_latency,
)
from fraud.training.tier1.search_results import logged_model_uri
from fraud.training.tier1.threshold import (
    ASSUMPTIONS,
    COST_RATIOS,
    DEFAULT_COST_RATIO,
    REVIEW_CAPACITY_FRACTION,
    STAGE_THRESHOLD,
    build_config,
    compute_thresholds,
    ensure_meaningful,
    print_sensitivity_table,
    threshold_metrics,
)
from fraud.training.tier2.tracking import setup_mlflow

ARM = "t1_plus"
CANDIDATE = "lightgbm"
THRESHOLDS_PATH = Path("configs/tier1_window_thresholds.json")
SMOKE_ROW_LIMIT = 30_000


def best_source_run(experiment_id: str) -> Run:
    """Pengulangan T1+ dengan PR-AUC validasi tertinggi; tiap seed diwakili run terbarunya.

    Raises:
        RuntimeError: Kalau belum ada pengulangan T1+ yang selesai.
    """
    runs = mlflow.MlflowClient().search_runs(
        [experiment_id],
        filter_string=(
            f"tags.arm = '{ARM}' and tags.run_role = 'trial' and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )
    latest: dict[str, Run] = {}
    for run in runs:
        latest.setdefault(run.data.tags["seed"], run)
    if not latest:
        raise RuntimeError("Belum ada pengulangan T1+ yang selesai di eksperimen ini.")
    return max(latest.values(), key=lambda run: run.data.metrics["validation_pr_auc"])


def _head(split: Split, rows: int) -> Split:
    return Split(
        features=split.features.iloc[:rows],
        label=split.label.iloc[:rows],
        entity_key=split.entity_key.iloc[:rows],
    )


def run_window_variant(smoke: bool, cost_ratio: int, thresholds_path: Path) -> None:
    """Mengalibrasi, menetapkan ambang, mengekspor, dan mendaftarkan T1+ terbaik.

    Raises:
        ValueError: Kalau skenario terpilih tidak bermakna atau ONNX tidak konsisten dengan model
            sklearn. Model tidak didaftarkan dalam kedua kasus.
    """
    experiment_id = setup_mlflow(smoke)
    source = best_source_run(experiment_id)
    seed = source.data.tags["seed"]
    print(
        f"Sumber: {ARM} seed {seed} (run {source.info.run_id[:8]}), "
        f"PR-AUC validasi {source.data.metrics['validation_pr_auc']:.4f}",
        flush=True,
    )
    model = mlflow_lightgbm.load_model(logged_model_uri(source))
    if not isinstance(model, LGBMClassifier):
        raise TypeError(f"Run {source.info.run_id} tidak mencatat LGBMClassifier.")

    data = prepare_datasets(with_window_features=True)
    train, validation, spec = data.train, data.validation, data.spec
    if smoke:
        train, validation = _head(train, SMOKE_ROW_LIMIT), _head(validation, SMOKE_ROW_LIMIT)
    source_params = {"source_run_id": source.info.run_id, "source_seed": seed, "arm": ARM}

    calibration = fit_isotonic(model, validation)
    print_calibration(calibration)
    score = calibration.model.predict_proba(validation.features)[:, 1]
    label = validation.label.to_numpy()
    results, selected = compute_thresholds(score, label, cost_ratio, REVIEW_CAPACITY_FRACTION)
    distinct_scores = len(np.unique(score))
    print_sensitivity_table(results, distinct_scores)
    # Diperiksa sebelum apa pun dicatat, supaya kegagalan tidak meninggalkan run kalibrasi yatim.
    ensure_meaningful(selected)

    with parent_run(CANDIDATE, STAGE_CALIBRATION, train, validation, spec) as run:
        mlflow.set_tag("arm", ARM)
        mlflow.log_params({**source_params, "method": CALIBRATION_METHOD})
        log_calibration(calibration)
        calibration_run_id = run.info.run_id
    config = build_config(
        results, selected, REVIEW_CAPACITY_FRACTION, CANDIDATE, calibration_run_id
    )
    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    print(f"Ambang ditulis ke {thresholds_path} (skenario {cost_ratio}:1)", flush=True)
    with parent_run(CANDIDATE, STAGE_THRESHOLD, train, validation, spec):
        mlflow.set_tag("arm", ARM)
        mlflow.log_params(
            {
                **source_params,
                "source_calibration_run_id": calibration_run_id,
                "selected_cost_ratio": cost_ratio,
                "review_capacity_fraction": REVIEW_CAPACITY_FRACTION,
                "cost_ratios": ",".join(str(ratio) for ratio in COST_RATIOS),
            }
        )
        mlflow.log_metrics(threshold_metrics(selected, distinct_scores))
        mlflow.log_dict(
            {"assumptions": list(ASSUMPTIONS), "scenarios": [asdict(r) for r in results]},
            "sensitivity_table.json",
        )
        mlflow.log_artifact(str(thresholds_path))

    n_features = len(spec["input_columns"])
    start = time.perf_counter()
    onnx_model, estimator = build_calibrated_onnx(calibration.model, n_features)
    model_bytes = onnx_model.SerializeToString()
    conversion_seconds = time.perf_counter() - start
    print(f"Konversi: {conversion_seconds:.0f} detik, {len(model_bytes) / 1e6:.1f} MB", flush=True)

    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    report = check_consistency(
        session, validation.features, calibration.model, estimator, config["scenarios"]
    )
    print(json.dumps(asdict(report), indent=2), flush=True)
    problems = report.failures()
    if problems:
        raise ValueError("ONNX tidak konsisten dengan model sklearn: " + "; ".join(problems))
    latency = measure_latency(model_bytes, validation.features)
    print("Latensi satu baris: " + ", ".join(f"{k} {v:.2f}" for k, v in latency.items()))

    registered_name = SMOKE_REGISTERED_MODEL_NAME if smoke else REGISTERED_MODEL_NAME
    with parent_run(CANDIDATE, STAGE_EXPORT, train, validation, spec):
        mlflow.set_tag("arm", ARM)
        log_and_register(
            onnx_model,
            spec,
            config,
            thresholds_path,
            calibration_run_id,
            CANDIDATE,
            registered_name,
            n_features,
        )
        mlflow.log_params(
            {
                **source_params,
                "source_calibration_run_id": calibration_run_id,
                "onnx_opset": ONNX_OPSET,
                "registered_model_name": registered_name,
            }
        )
        mlflow.log_metrics(export_metrics(report, latency, model_bytes, conversion_seconds))
        mlflow.log_dict(asdict(report), "consistency_report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daftarkan T1+ sebagai versi baru tier 1.")
    parser.add_argument(
        "--cost-ratio",
        type=int,
        choices=COST_RATIOS,
        default=DEFAULT_COST_RATIO,
        help="Skenario rasio biaya (lolos-fraud : salah-tolak) yang dipakai sebagai titik operasi.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Uji cepat: data dipotong, eksperimen dan registry smoke, ambang ke folder sementara.",
    )
    args = parser.parse_args()
    if args.smoke:
        # Ambang smoke tidak boleh menimpa konfigurasi yang ikut dikemas bersama model asli.
        with tempfile.TemporaryDirectory() as directory:
            run_window_variant(True, args.cost_ratio, Path(directory) / THRESHOLDS_PATH.name)
    else:
        run_window_variant(False, args.cost_ratio, THRESHOLDS_PATH)


if __name__ == "__main__":
    main()
