"""Strategi pencarian hyperparameter tier 1: phase 1 (Optuna), phase 2 (grid lokal), konfirmasi.

Anggaran pencarian ditentukan oleh peran kandidat, bukan oleh peringkat baseline: kandidat
utama dicari lebih dalam, pembanding secukupnya supaya perbandingannya tetap adil. Tiap tahap
dijalankan sebagai perintah terpisah dan saling membaca hasil lewat MLflow, bukan lewat memori,
sehingga bisa dijalankan di sesi terminal yang berbeda.

Dijalankan lewat:
    uv run python -m fraud.training.tier1.search --stage {search,refine,confirm,report} [...]
"""

import argparse
import itertools
import math
import statistics
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

import mlflow
import optuna
from mlflow.entities import Run

from fraud.training.tier1.candidates import (
    CANDIDATE_NAMES,
    MAIN_CANDIDATES,
    OPERATIONAL_PARAMS,
    RANDOM_SEED,
    STAGE_BASELINE,
    STAGE_CONFIRM,
    STAGE_PHASE1,
    STAGE_PHASE2,
    TrialResult,
    load_training_data,
    log_best,
    parent_run,
    run_trial,
    setup_mlflow,
)
from fraud.training.tier1.dataset import FeatureSpec, Split
from fraud.training.tier1.search_results import (
    best_parent as _best_parent,
    confirmation_of,
    finished_parents as _finished_parents,
    load_run_dict as _load_run_dict,
)


@dataclass(frozen=True)
class NumericParam:
    """Parameter numerik yang dicari Optuna."""

    name: str
    low: float
    high: float
    is_integer: bool = False
    log_scale: bool = False


@dataclass(frozen=True)
class CategoricalParam:
    """Parameter dengan daftar pilihan diskret yang dicari Optuna."""

    name: str
    choices: tuple


SearchParam = NumericParam | CategoricalParam

# Ronde 2: digeser mengikuti sebaran trial teratas ronde 1, bukan sekadar dilebarkan.
SEARCH_SPACES: dict[str, list[SearchParam]] = {
    "random_forest": [
        CategoricalParam("max_depth", (20, 30, 40, None)),
        NumericParam("min_samples_leaf", 1, 30, is_integer=True, log_scale=True),
        NumericParam("max_features", 0.03, 0.3, log_scale=True),
    ],
    "lightgbm": [
        # Ronde 1: trial di bawah 100 daun paling bagus hanya 0,6106, jauh di luar sepuluh besar.
        NumericParam("num_leaves", 100, 1500, is_integer=True, log_scale=True),
        NumericParam("min_child_samples", 5, 200, is_integer=True, log_scale=True),
        NumericParam("learning_rate", 0.01, 0.08, log_scale=True),
        NumericParam("n_estimators", 400, 3000, is_integer=True, log_scale=True),
        NumericParam("colsample_bytree", 0.3, 1.0),
        # Ronde 2: subsample di bawah 0,6 skor terbaiknya cuma 0,6317, jauh di luar sepuluh besar.
        NumericParam("subsample", 0.7, 1.0),
        NumericParam("reg_lambda", 0.001, 10, log_scale=True),
    ],
    "xgboost": [
        NumericParam("max_depth", 8, 16, is_integer=True),
        NumericParam("min_child_weight", 0.1, 20, log_scale=True),
        # Default 0,3 sengaja di luar rentang: dirancang untuk 100 putaran, di sini ratusan pohon.
        NumericParam("learning_rate", 0.01, 0.08, log_scale=True),
        # Ronde 1: trial dengan 349 pohon masih di peringkat 15, jadi batas bawah tidak di 350.
        NumericParam("n_estimators", 300, 3000, is_integer=True, log_scale=True),
        NumericParam("colsample_bytree", 0.3, 1.0),
        NumericParam("subsample", 0.5, 1.0),
        NumericParam("reg_lambda", 0.001, 10, log_scale=True),
    ],
}

# Regresi logistik hanya punya satu parameter, jadi disapu berurutan, tanpa Optuna.
LOGISTIC_REGRESSION_C_VALUES = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)

# Parameter yang tidak dicari tapi berbeda dari bawaan library, dengan alasan masing-masing.
SEARCH_FIXED_PARAMS: dict[str, dict[str, Any]] = {
    # Cukup besar supaya konvergen, bukan hyperparameter kualitas.
    "logistic_regression": {"max_iter": 2000},
    # Lebih banyak pohon hanya menurunkan varians lalu mendatar, menambah waktu tanpa menambah skor.
    "random_forest": {"n_estimators": 300},
    # Tanpa bagging_freq, subsample diabaikan. num_leaves sudah membatasi kedalaman pohon.
    "lightgbm": {"subsample_freq": 1, "max_depth": -1},
    "xgboost": {"tree_method": "hist"},
}

# Batas atas, pencarian GBM berhenti lebih awal kalau sudah mendatar (lihat aturan di bawah).
SEARCH_TRIALS = {"lightgbm": 60, "xgboost": 60, "random_forest": 15}
SMOKE_TRIALS = 5
MAX_STARTUP_TRIALS = 10

# Berhenti kalau PATIENCE trial terakhir tidak menaikkan skor terbaik lebih dari MIN_GAIN.
# MIN_GAIN kira-kira sebesar noise antar trial, dan MIN_TRIALS mencegah berhenti sebelum
# TPE sempat terpandu.
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_GAIN = 0.002
EARLY_STOP_MIN_TRIALS = 25

# Jumlah konfigurasi terbaik ronde sebelumnya yang dilatih ulang paling awal di study baru.
WARM_START_TRIALS = 3

# Parameter dianggap menempel di tepi kalau sebanyak ini dari sepuluh trial teratas berada di
# sepuluh persen terluar rentangnya (skala log untuk parameter berskala log).
EDGE_FRACTION = 0.1
EDGE_MIN_COUNT = 3
EDGE_TOP_TRIALS = 10

REFINED_PARAMETER_COUNT = 3
# Langkah grid lokal: perkalian untuk parameter berskala log, penambahan untuk fraksi.
LOG_STEP = 1.5
FRACTION_STEP = 0.1
MIN_FRACTION = 0.1
MAX_FRACTION = 1.0

CONFIRMATION_SEEDS = (RANDOM_SEED, RANDOM_SEED + 1, RANDOM_SEED + 2)

# Pembanding yang skornya sedekat ini dari yang terbaik layak ikut phase 2.
PROMOTION_MARGIN = 0.01

REFINABLE_CANDIDATES = ("random_forest", "lightgbm", "xgboost")


def _merged_params(name: str, tuned: dict[str, Any]) -> dict[str, Any]:
    """Menggabungkan parameter operasional, parameter tetap, dan parameter yang dicari."""
    return {**OPERATIONAL_PARAMS[name], **SEARCH_FIXED_PARAMS[name], **tuned}


def _keep_better(best: TrialResult | None, result: TrialResult) -> TrialResult:
    """Mengembalikan percobaan dengan PR-AUC validasi lebih tinggi."""
    if best is None or result.validation_pr_auc > best.validation_pr_auc:
        return result
    return best


def _suggest(trial: optuna.Trial, param: SearchParam) -> Any:
    """Meminta Optuna memilih nilai satu parameter untuk trial ini."""
    if isinstance(param, CategoricalParam):
        return trial.suggest_categorical(param.name, list(param.choices))
    if param.is_integer:
        return trial.suggest_int(param.name, int(param.low), int(param.high), log=param.log_scale)
    return trial.suggest_float(param.name, param.low, param.high, log=param.log_scale)


def _param_importances(study: optuna.Study) -> dict[str, float]:
    """Kepentingan tiap parameter terhadap skor, urut dari yang terbesar."""
    return dict(optuna.importance.get_param_importances(study))


def _stop_when_plateaued(
    study: optuna.Study, _trial: optuna.trial.FrozenTrial, warm_start_count: int = 0
) -> None:
    """Callback Optuna: menghentikan study kalau skor terbaik sudah mendatar.

    Args:
        warm_start_count: Jumlah trial pertama yang merupakan replay konfigurasi lama
            (warm start), dikecualikan dari perhitungan plateau. Tanpa ini, replay
            konfigurasi yang sudah bagus langsung mengisi baseline pembanding, sehingga
            plateau terdeteksi meski trial baru belum sempat menjelajah ruang pencarian.
    """
    scores = [t.value for t in study.trials if t.value is not None][warm_start_count:]
    if len(scores) < EARLY_STOP_MIN_TRIALS:
        return
    earlier, recent = scores[:-EARLY_STOP_PATIENCE], scores[-EARLY_STOP_PATIENCE:]
    if max(recent) < max(earlier) + EARLY_STOP_MIN_GAIN:
        study.stop()


def _warm_start_configs(
    experiment_id: str, name: str, space: list[SearchParam], count: int
) -> list[dict[str, Any]]:
    """Konfigurasi terbaik phase 1 sebelumnya yang masih berada di dalam ruang pencarian ini.

    Args:
        experiment_id: Eksperimen MLflow tempat trial lama dibaca.
        name: Kandidat.
        space: Ruang pencarian baru. Konfigurasi lama yang sebagian nilainya di luar ruang ini
            dilewati, karena Optuna tidak bisa menjalankannya di ruang ini.
        count: Jumlah konfigurasi yang diambil, dari skor tertinggi.

    Returns:
        Sampai `count` konfigurasi, tiap konfigurasi berupa nilai untuk seluruh parameter di
        `space`. Kosong kalau ruangnya memuat parameter kategorikal.
    """
    numeric_space = [param for param in space if isinstance(param, NumericParam)]
    if len(numeric_space) != len(space):
        return []
    old_trials = mlflow.MlflowClient().search_runs(
        [experiment_id],
        filter_string=(
            f"tags.candidate = '{name}' and tags.run_role = 'trial' "
            f"and tags.search_stage = '{STAGE_PHASE1}' and attributes.status = 'FINISHED'"
        ),
        order_by=["metrics.validation_pr_auc DESC"],
        max_results=500,
    )
    configs: list[dict[str, Any]] = []
    for old_trial in old_trials:
        values = {}
        for param in numeric_space:
            raw = old_trial.data.params.get(param.name)
            if raw is None:
                break
            values[param.name] = int(float(raw)) if param.is_integer else float(raw)
        else:
            if all(param.low <= values[param.name] <= param.high for param in numeric_space):
                configs.append(values)
        if len(configs) == count:
            break
    return configs


def run_search(
    experiment_id: str,
    name: str,
    n_trials: int,
    train: Split,
    validation: Split,
    spec: FeatureSpec,
    warm_start: bool = False,
) -> TrialResult:
    """Menjalankan phase 1: pencarian Optuna (TPE) pada ruang parameter satu kandidat.

    Args:
        experiment_id: Eksperimen MLflow, dibaca untuk warm start.
        name: Kandidat yang dicari, kunci di `SEARCH_SPACES`.
        n_trials: Batas atas jumlah percobaan. Pencarian bisa berhenti lebih awal kalau skor
            terbaik sudah mendatar.
        train: Split train.
        validation: Split validasi.
        spec: Kontrak input model.
        warm_start: Kalau True, konfigurasi terbaik phase 1 sebelumnya yang masih di dalam
            ruang pencarian ini dilatih ulang lebih dulu sebagai titik awal TPE.

    Returns:
        Percobaan terbaik. Seluruh percobaan, kepentingan parameter, dan model terbaik
        tercatat di MLflow.
    """
    space = SEARCH_SPACES[name]
    n_startup_trials = min(MAX_STARTUP_TRIALS, max(1, n_trials // 3))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_SEED, multivariate=True, n_startup_trials=n_startup_trials
        ),
    )
    warm_configs: list[dict[str, Any]] = []
    if warm_start:
        warm_configs = _warm_start_configs(experiment_id, name, space, WARM_START_TRIALS)
        print(f"  warm start: {len(warm_configs)} konfigurasi ronde sebelumnya", flush=True)
    for config in warm_configs:
        study.enqueue_trial(config)

    best: TrialResult | None = None
    with parent_run(name, STAGE_PHASE1, train, validation, spec):
        mlflow.log_params(
            {
                "n_trials": n_trials,
                "n_startup_trials": n_startup_trials,
                "warm_start_trials": len(warm_configs),
            }
        )
        mlflow.log_dict({"parameters": [asdict(param) for param in space]}, "search_space.json")

        def objective(trial: optuna.Trial) -> float:
            nonlocal best
            tuned = {param.name: _suggest(trial, param) for param in space}
            params = _merged_params(name, tuned)
            result = run_trial(name, STAGE_PHASE1, trial.number, params, train, validation, spec)
            best = _keep_better(best, result)
            return result.validation_pr_auc

        stop_callback = partial(_stop_when_plateaued, warm_start_count=len(warm_configs))
        study.optimize(objective, n_trials=n_trials, callbacks=[stop_callback])
        completed = len([t for t in study.trials if t.value is not None])
        mlflow.log_metric("completed_trials", completed)
        mlflow.set_tag("stopped_early", str(completed < n_trials))
        mlflow.log_dict(_param_importances(study), "param_importances.json")
        assert best is not None
        log_best(name, best)
    return best


def run_logistic_regression_sweep(
    train: Split, validation: Split, spec: FeatureSpec
) -> TrialResult:
    """Menjalankan phase 1 regresi logistik: sapuan berurutan atas nilai `C`."""
    name = "logistic_regression"
    best: TrialResult | None = None
    with parent_run(name, STAGE_PHASE1, train, validation, spec):
        mlflow.log_dict({"C": list(LOGISTIC_REGRESSION_C_VALUES)}, "search_space.json")
        for index, c_value in enumerate(LOGISTIC_REGRESSION_C_VALUES):
            params = _merged_params(name, {"C": c_value})
            best = _keep_better(
                best, run_trial(name, STAGE_PHASE1, index, params, train, validation, spec)
            )
        assert best is not None
        log_best(name, best)
    return best


def _round_significant(value: float) -> float:
    """Membulatkan ke 6 angka signifikan supaya nilai grid tidak berekor panjang."""
    return float(f"{value:.6g}")


def _neighbor_values(param: NumericParam, value: float) -> list[float]:
    """Tiga titik grid lokal berpusat pada `value`: satu langkah ke bawah dan ke atas."""
    if param.log_scale:
        low, high = value / LOG_STEP, value * LOG_STEP
    elif param.is_integer:
        low, high = value - 1, value + 1
    else:
        # Satu-satunya parameter float linier adalah fraksi baris atau fitur, sah di 0,1 sampai 1.
        low, high = (
            min(max(step_value, MIN_FRACTION), MAX_FRACTION)
            for step_value in (value - FRACTION_STEP, value + FRACTION_STEP)
        )
    if param.is_integer:
        low, high = max(round(low), 1), max(round(high), 1)
    else:
        low, high = _round_significant(low), _round_significant(high)
    return sorted({low, value, high})


def refinement_axes(
    space: list[SearchParam], best_params: dict[str, Any], importances: dict[str, float]
) -> dict[str, list[float]]:
    """Memilih parameter terpenting dan titik grid lokalnya untuk phase 2.

    Args:
        space: Ruang pencarian phase 1.
        best_params: Parameter lengkap percobaan terbaik phase 1.
        importances: Kepentingan parameter dari phase 1, urut dari yang terbesar.

    Returns:
        Titik grid per parameter. Parameter kategorikal tidak ikut dihaluskan.
    """
    numeric = {param.name: param for param in space if isinstance(param, NumericParam)}
    ranked = [name for name in importances if name in numeric][:REFINED_PARAMETER_COUNT]
    return {name: _neighbor_values(numeric[name], best_params[name]) for name in ranked}


def grid_combinations(
    axes: dict[str, list[float]], best_params: dict[str, Any]
) -> list[dict[str, float]]:
    """Seluruh kombinasi grid lokal, kecuali titik pusat yang sudah dijalankan di phase 1."""
    center = {name: best_params[name] for name in axes}
    combinations = [dict(zip(axes, values)) for values in itertools.product(*axes.values())]
    return [combination for combination in combinations if combination != center]


def _load_logged_space(run: Run) -> list[SearchParam]:
    """Ruang pencarian yang benar-benar dipakai sebuah run, dibaca dari artefaknya.

    Dibaca dari run, bukan dari `SEARCH_SPACES`, karena isi kode bisa sudah berubah sejak run
    itu berjalan.
    """
    entries = _load_run_dict(run, "search_space.json")["parameters"]
    return [
        CategoricalParam(entry["name"], tuple(entry["choices"]))
        if "choices" in entry
        else NumericParam(**entry)
        for entry in entries
    ]


def _position_in_range(param: NumericParam, value: float) -> float:
    """Posisi nilai di rentang parameter: 0 di batas bawah, 1 di batas atas."""
    if param.log_scale:
        return math.log(value / param.low) / math.log(param.high / param.low)
    return (value - param.low) / (param.high - param.low)


def edge_findings(space: list[SearchParam], top_params: list[dict[str, float]]) -> list[str]:
    """Parameter yang nilai terbaiknya menempel di tepi rentang pencarian.

    Args:
        space: Ruang pencarian yang dipakai run. Parameter kategorikal dilewati.
        top_params: Nilai parameter pada trial-trial teratas, satu dict per trial.

    Returns:
        Satu kalimat temuan per parameter yang setidaknya `EDGE_MIN_COUNT` trial teratasnya
        berada di `EDGE_FRACTION` terluar rentang di sisi yang sama.
    """
    findings = []
    for param in space:
        if not isinstance(param, NumericParam):
            continue
        positions = [_position_in_range(param, trial[param.name]) for trial in top_params]
        at_top = sum(position > 1 - EDGE_FRACTION for position in positions)
        at_bottom = sum(position < EDGE_FRACTION for position in positions)
        for count, side in ((at_top, "atas"), (at_bottom, "bawah")):
            if count >= EDGE_MIN_COUNT:
                findings.append(
                    f"{param.name}: {count} dari {len(positions)} trial teratas di tepi {side} "
                    f"rentang [{param.low:g}, {param.high:g}]"
                )
    return findings


def _phase1_edge_findings(run: Run) -> list[str]:
    """Temuan tepi untuk sebuah run induk phase 1, dari trial teratasnya."""
    space = _load_logged_space(run)
    names = [param.name for param in space if isinstance(param, NumericParam)]
    top_trials = mlflow.MlflowClient().search_runs(
        [run.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{run.info.run_id}'",
        order_by=["metrics.validation_pr_auc DESC"],
        max_results=EDGE_TOP_TRIALS,
    )
    top_params = [{name: float(t.data.params[name]) for name in names} for t in top_trials]
    return edge_findings(space, top_params)


def run_refinement(
    experiment_id: str,
    name: str,
    train: Split,
    validation: Split,
    spec: FeatureSpec,
    allow_edge: bool = False,
) -> TrialResult:
    """Menjalankan phase 2: grid lokal di sekitar hasil terbaik phase 1.

    Args:
        experiment_id: Eksperimen MLflow tempat hasil phase 1 dibaca.
        name: Kandidat, harus sudah punya run phase 1 yang selesai.
        train: Split train.
        validation: Split validasi.
        spec: Kontrak input model.
        allow_edge: Kalau False, phase 2 menolak berjalan selama masih ada parameter yang
            menempel di tepi rentang phase 1.

    Returns:
        Percobaan terbaik di antara titik grid yang baru dijalankan.

    Raises:
        RuntimeError: Kalau phase 1 belum selesai, tidak punya kepentingan parameter, atau
            masih ada parameter di tepi dan `allow_edge` False.
    """
    phase1 = _best_parent(experiment_id, name, (STAGE_PHASE1,))
    best_params = _load_run_dict(phase1, "best_params.json")
    importances = _load_run_dict(phase1, "param_importances.json")
    if not importances:
        raise RuntimeError(f"Phase 1 {name} tidak punya kepentingan parameter, ulangi phase 1.")

    findings = _phase1_edge_findings(phase1)
    if findings and not allow_edge:
        details = "\n  ".join(findings)
        raise RuntimeError(
            f"Phase 1 {name} masih menempel di tepi rentang:\n  {details}\n"
            "Perluas rentang dan ulangi phase 1, atau jalankan dengan --allow-edge."
        )
    for finding in findings:
        print(f"  PERINGATAN tepi (dilanjutkan karena --allow-edge): {finding}", flush=True)

    axes = refinement_axes(_load_logged_space(phase1), best_params, importances)
    combinations = grid_combinations(axes, best_params)

    best: TrialResult | None = None
    with parent_run(name, STAGE_PHASE2, train, validation, spec):
        mlflow.log_params(
            {
                "source_phase1_run_id": phase1.info.run_id,
                "phase1_best_pr_auc": round(phase1.data.metrics["best_validation_pr_auc"], 4),
                "refined_parameters": ",".join(axes),
                "trial_count": len(combinations),
            }
        )
        mlflow.log_dict(axes, "refinement_axes.json")
        for index, combination in enumerate(combinations):
            params = {**best_params, **combination}
            best = _keep_better(
                best, run_trial(name, STAGE_PHASE2, index, params, train, validation, spec)
            )
        assert best is not None
        log_best(name, best)
    return best


def run_confirmation(
    experiment_id: str, name: str, train: Split, validation: Split, spec: FeatureSpec
) -> list[float]:
    """Mengulang konfigurasi terbaik dengan beberapa seed untuk mengukur noise skornya.

    Args:
        experiment_id: Eksperimen MLflow tempat hasil pencarian dibaca.
        name: Kandidat, harus sudah punya hasil phase 1 atau phase 2.
        train: Split train.
        validation: Split validasi.
        spec: Kontrak input model.

    Returns:
        PR-AUC validasi tiap seed di `CONFIRMATION_SEEDS`.
    """
    source = _best_parent(experiment_id, name, (STAGE_PHASE1, STAGE_PHASE2))
    best_params = _load_run_dict(source, "best_params.json")

    scores = []
    with parent_run(name, STAGE_CONFIRM, train, validation, spec):
        mlflow.log_params(
            {
                "source_run_id": source.info.run_id,
                "source_stage": source.data.tags["search_stage"],
                "seeds": ",".join(str(seed) for seed in CONFIRMATION_SEEDS),
            }
        )
        for index, seed in enumerate(CONFIRMATION_SEEDS):
            params = {**best_params, "random_state": seed}
            result = run_trial(name, STAGE_CONFIRM, index, params, train, validation, spec)
            scores.append(result.validation_pr_auc)
        mlflow.log_metrics(
            {
                "pr_auc_mean": statistics.mean(scores),
                "pr_auc_std": statistics.stdev(scores),
                "pr_auc_min": min(scores),
                "pr_auc_max": max(scores),
            }
        )
    return scores


def _format_score(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def print_edge_check(experiment_id: str) -> None:
    """Mencetak parameter yang menempel di tepi rentang pada phase 1 terbaik tiap kandidat."""
    print("\nPemeriksaan tepi rentang phase 1:")
    for name in REFINABLE_CANDIDATES:
        try:
            run = _best_parent(experiment_id, name, (STAGE_PHASE1,))
        except RuntimeError:
            continue
        findings = _phase1_edge_findings(run)
        if not findings:
            print(f"  {name}: tidak ada parameter di tepi")
        for finding in findings:
            print(f"  {name}: TEPI  {finding}")


def print_report(experiment_id: str) -> None:
    """Merangkum skor tiap kandidat per tahap dari MLflow, plus kandidat yang layak phase 2."""
    print(
        f"{'kandidat':22s} {'baseline':>9s} {'phase 1':>9s} {'phase 2':>9s}  "
        f"{'konfirmasi phase 1':>19s} {'konfirmasi phase 2':>19s}"
    )
    best_scores: dict[str, float] = {}
    refined: set[str] = set()
    for name in CANDIDATE_NAMES:
        latest: dict[str, Run] = {}
        for run in _finished_parents(experiment_id, name):
            latest.setdefault(run.data.tags.get("search_stage", ""), run)

        def score(stage: str) -> float | None:
            run = latest.get(stage)
            return run.data.metrics["best_validation_pr_auc"] if run else None

        def confirmation_text(stage: str) -> str:
            run = latest.get(stage)
            confirmation = confirmation_of(experiment_id, name, run.info.run_id) if run else None
            if confirmation is None:
                return "-"
            return f"{confirmation.mean:.4f} +/- {confirmation.std:.4f}"

        searched = [s for s in (score(STAGE_PHASE1), score(STAGE_PHASE2)) if s is not None]
        if searched:
            best_scores[name] = max(searched)
        if STAGE_PHASE2 in latest:
            refined.add(name)
        print(
            f"{name:22s} {_format_score(score(STAGE_BASELINE)):>9s} "
            f"{_format_score(score(STAGE_PHASE1)):>9s} {_format_score(score(STAGE_PHASE2)):>9s}  "
            f"{confirmation_text(STAGE_PHASE1):>19s} {confirmation_text(STAGE_PHASE2):>19s}"
        )

    print_edge_check(experiment_id)

    if not best_scores:
        return
    top = max(best_scores.values())
    promoted = [
        name
        for name, value in best_scores.items()
        if name in REFINABLE_CANDIDATES
        and name not in MAIN_CANDIDATES
        and name not in refined
        and top - value <= PROMOTION_MARGIN
    ]
    if promoted:
        names = ", ".join(promoted)
        print(f"\nLayak phase 2 (dalam {PROMOTION_MARGIN} dari yang terbaik): {names}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pencarian hyperparameter tier 1.")
    parser.add_argument("--stage", required=True, choices=("search", "refine", "confirm", "report"))
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=CANDIDATE_NAMES,
        help="Kandidat yang diproses. Default: semua untuk search, dua utama untuk lainnya.",
    )
    parser.add_argument(
        "--n-trials", type=int, help="Ganti jumlah trial Optuna, hanya untuk stage search."
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Stage search: latih ulang konfigurasi terbaik ronde sebelumnya lebih dulu.",
    )
    parser.add_argument(
        "--allow-edge",
        action="store_true",
        help="Stage refine: tetap jalan walau masih ada parameter di tepi rentang phase 1.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Uji cepat: data dipotong dan dicatat di eksperimen terpisah.",
    )
    args = parser.parse_args()

    experiment_id = setup_mlflow(args.smoke)
    if args.stage == "report":
        print_report(experiment_id)
        return

    if args.stage == "search":
        candidates = args.candidates or list(CANDIDATE_NAMES)
    else:
        candidates = args.candidates or list(MAIN_CANDIDATES)
        unsupported = [name for name in candidates if name not in REFINABLE_CANDIDATES]
        if unsupported:
            parser.error(f"{args.stage} tidak berlaku untuk {', '.join(unsupported)}.")

    train, validation, spec = load_training_data(args.smoke)
    for name in candidates:
        print(f"{args.stage} {name} ...", flush=True)
        if args.stage == "search" and name == "logistic_regression":
            result = run_logistic_regression_sweep(train, validation, spec)
        elif args.stage == "search":
            n_trials = args.n_trials or (SMOKE_TRIALS if args.smoke else SEARCH_TRIALS[name])
            result = run_search(
                experiment_id, name, n_trials, train, validation, spec, args.warm_start
            )
        elif args.stage == "refine":
            result = run_refinement(
                experiment_id, name, train, validation, spec, args.allow_edge
            )
        else:
            scores = run_confirmation(experiment_id, name, train, validation, spec)
            print(f"  skor per seed: {[round(score, 4) for score in scores]}")
            continue
        print(f"  terbaik: PR-AUC {result.validation_pr_auc:.4f}  {result.run_name}")


if __name__ == "__main__":
    main()
