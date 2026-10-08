"""Evaluasi pengujian hipotesis tier 2 dengan kriteria sukses yang ditetapkan sebelum pelatihan.

Lima lengan dibandingkan pada transaksi yang sama: T1 dan T1+ (tier 1 tanpa dan dengan fitur
jendela waktu) serta tiga lengan tier 2 (`t2_ordered`, `t2_shuffled`, `t2_current_only`). Setiap
lengan punya beberapa pengulangan dengan seed berbeda. Sebuah keunggulan dianggap terdukung hanya
kalau lolos dua gerbang sekaligus:

1. Variasi pelatihan: selisih rata-rata PR-AUC lebih dari `NOISE_MULTIPLIER` kali galat baku
   selisih, dihitung dari simpangan baku antar seed kedua lengan.
2. Variasi data evaluasi: selang kepercayaan selisih dari bootstrap berpasangan per blok hari
   seluruhnya di atas nol. Hari menjadi unit resampel karena transaksi pada hari yang sama saling
   berkorelasi, dan pergeseran pola antar periode sudah terbukti pada dataset ini.

Lolos kedua gerbang ke arah sebaliknya berarti berlawanan; selain itu tidak meyakinkan. Kesimpulan
diambil dari hasil global; hasil per segmen ukuran entitas dibaca sebagai indikasi, karena enam
perbandingan segmen tanpa koreksi memperbesar peluang hasil terdukung yang kebetulan.

Mode `validation` boleh dijalankan berulang dan memuat putusan promosi T1+ terhadap T1. Mode
`test` hanya boleh dijalankan sekali untuk seluruh paket lengan, setelah semua pengulangan selesai.

Dijalankan lewat:
    uv run python -m fraud.training.tier2.evaluate --split {validation,test} [--smoke]
"""

import argparse
import json
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import mlflow.lightgbm as mlflow_lightgbm
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from matplotlib.figure import Figure
from mlflow.entities import Run
from sklearn.metrics import average_precision_score, precision_recall_curve

from fraud.features.offline_store import INITIAL_PARQUET_DIR, SECONDS_PER_DAY
from fraud.training.tier1.dataset import prepare_datasets
from fraud.training.tier1.evaluate import (
    POSITION_SEGMENTS,
    SIZE_SEGMENTS,
    position_segment_keys,
    size_segment_keys,
)
from fraud.training.tier1.fair_comparator import ARMS as TIER1_ARMS
from fraud.training.tier1.search_results import logged_model_uri
from fraud.training.tier2.ablation_shuffle import arm_seq_len, latest_main_runs
from fraud.training.tier2.dataset import SequenceData, prepare_sequence_data
from fraud.training.tier2.finetune import (
    CHECKPOINT_ARTIFACT as FINETUNE_CHECKPOINT_ARTIFACT,
    ensure_same_spec,
    load_run_package,
    predict,
    selected_configuration,
)
from fraud.training.tier2.model import Tier2Model
from fraud.training.tier2.pretrain import SMOKE_ROW_LIMIT
from fraud.training.tier2.tracking import (
    HYPOTHESIS_SEEDS,
    select_device,
    setup_mlflow,
)

STAGE = "evaluation"
SPLITS = ("validation", "test")

# Kriteria sukses, ditetapkan dan disetujui sebelum pelatihan apa pun dijalankan.
NOISE_MULTIPLIER = 2.0
BOOTSTRAP_RESAMPLES = 1000
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SEED = 42
REQUIRED_SEEDS = HYPOTHESIS_SEEDS
SMOKE_SEED_COUNT = 2

SUPPORTED = "terdukung"
OPPOSED = "berlawanan"
INCONCLUSIVE = "tidak meyakinkan"
VERDICT_CODES = {SUPPORTED: 1, OPPOSED: -1, INCONCLUSIVE: 0}

TIER2_ARMS = ("t2_ordered", "t2_shuffled", "t2_current_only")
ARM_ORDER = (*TIER1_ARMS, *TIER2_ARMS)

# Hasil uji tier 1 produksi versi 1, sudah dilaporkan sekali dan hanya dikutip sebagai acuan.
REFERENCE_TIER1_RUN_ID = "b4861e6a162a43648967cb4f442bac0c"

# Warna mengikuti lengan, bukan peringkat, sehingga gambar antar evaluasi tetap sebanding.
ARM_COLORS = dict(zip(ARM_ORDER, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")))
SURFACE_COLOR = "#fcfcfb"
PR_CURVE_POINTS = 2000


@dataclass(frozen=True)
class ComparisonSpec:
    """Satu perbandingan antar lengan.

    Attributes:
        name: Pengenal perbandingan untuk metrik dan laporan.
        challenger: Lengan yang diuji lebih baik.
        baseline: Lengan pembanding.
        role: `promotion`, `decisive` (penentu pengujian), `control` (penafsiran), atau
            `diagnostic` (perbandingan pengujian yang dihitung di validasi).
        by_size: Apakah juga dihitung per segmen ukuran entitas.
    """

    name: str
    challenger: str
    baseline: str
    role: str
    by_size: bool = False


PROMOTION = ComparisonSpec("promotion_t1_plus_vs_t1", "t1_plus", "t1", "promotion")
HYPOTHESIS_COMPARISONS = (
    ComparisonSpec("test1_ordered_vs_t1_plus", "t2_ordered", "t1_plus", "decisive", by_size=True),
    ComparisonSpec("test2_ordered_vs_shuffled", "t2_ordered", "t2_shuffled", "decisive", True),
    ComparisonSpec("control_ordered_vs_current_only", "t2_ordered", "t2_current_only", "control"),
    ComparisonSpec("control_current_only_vs_t1_plus", "t2_current_only", "t1_plus", "control"),
)


@dataclass(frozen=True)
class ArmScores:
    """Skor seluruh pengulangan satu lengan pada baris evaluasi yang sama.

    Attributes:
        arm: Nama lengan.
        seeds: Seed tiap pengulangan.
        run_ids: Run sumber model tiap pengulangan.
        scores: Skor `(jumlah seed, jumlah baris)`, urutan baris sesuai `EvaluationRows`.
    """

    arm: str
    seeds: list[int]
    run_ids: list[str]
    scores: np.ndarray


@dataclass(frozen=True)
class EvaluationRows:
    """Transaksi yang dinilai pada split evaluasi, beserta atribut pengelompokannya."""

    transaction_id: np.ndarray
    label: np.ndarray
    day_index: np.ndarray
    n_days: int
    size_key: np.ndarray
    position_key: np.ndarray


@dataclass(frozen=True)
class RankedScores:
    """Skor satu pengulangan yang sudah diurutkan sekali, siap untuk PR-AUC berbobot berulang.

    Attributes:
        label: Label urut skor menurun.
        day_index: Indeks hari urut skor menurun.
        group_end: True di elemen terakhir setiap kelompok skor kembar.
    """

    label: np.ndarray
    day_index: np.ndarray
    group_end: np.ndarray


@dataclass(frozen=True)
class ComparisonResult:
    """Hasil satu perbandingan pada satu segmen, beserta kedua gerbangnya."""

    name: str
    role: str
    segment: str
    challenger: str
    baseline: str
    n_rows: int
    n_fraud: int
    challenger_mean: float
    challenger_std: float | None
    baseline_mean: float
    baseline_std: float | None
    margin: float
    standard_error: float | None
    ci_low: float | None
    ci_high: float | None
    valid_resamples: int
    verdict: str


def rank_scores(score: np.ndarray, label: np.ndarray, day_index: np.ndarray) -> RankedScores:
    """Mengurutkan skor menurun sekali; urutan stabil supaya hasil selalu sama."""
    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    group_end = np.ones(len(order), dtype=bool)
    group_end[:-1] = sorted_score[1:] != sorted_score[:-1]
    return RankedScores(label=label[order], day_index=day_index[order], group_end=group_end)


def weighted_average_precision(ranked: RankedScores, day_weights: np.ndarray) -> float:
    """PR-AUC (average precision) dengan bobot per hari, setara `average_precision_score`.

    Resampel bootstrap per blok hari diwujudkan sebagai bobot: berapa kali hari baris itu
    terpilih. Skor kembar diperlakukan sebagai satu ambang, sama seperti scikit-learn.

    Returns:
        PR-AUC, atau NaN kalau bobot total fraud nol.
    """
    weights = day_weights[ranked.day_index]
    true_positive = np.cumsum(weights * ranked.label)[ranked.group_end]
    false_positive = np.cumsum(weights * (1 - ranked.label))[ranked.group_end]
    total_positive = true_positive[-1]
    if total_positive <= 0:
        return float("nan")
    predicted = true_positive + false_positive
    precision = np.divide(
        true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0
    )
    recall = true_positive / total_positive
    return float(np.sum(np.diff(recall, prepend=0.0) * precision))


def check_average_precision() -> None:
    """Membuktikan PR-AUC berbobot setara scikit-learn sebelum dipakai untuk putusan.

    Diuji pada data sintetis dengan skor kembar dan hari berbobot nol.

    Raises:
        RuntimeError: Ada kasus yang tidak setara.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for case in range(50):
        n_rows, n_days = 500, 20
        # Pembulatan ke satu desimal memaksa banyak skor kembar.
        score = np.round(rng.random(n_rows), 1 if case % 2 else 4)
        label = (rng.random(n_rows) < 0.1).astype(float)
        label[0] = 1.0
        day_index = rng.integers(0, n_days, n_rows)
        day_weights = rng.integers(0, 3, n_days).astype(float)
        day_weights[day_index[0]] = 1.0
        expected = average_precision_score(label, score, sample_weight=day_weights[day_index])
        actual = weighted_average_precision(rank_scores(score, label, day_index), day_weights)
        if not np.isclose(actual, expected, rtol=0, atol=1e-12):
            raise RuntimeError(f"PR-AUC berbobot tidak setara scikit-learn: {actual} vs {expected}")


def verdict(
    margin: float, standard_error: float | None, ci_low: float | None, ci_high: float | None
) -> str:
    """Putusan satu perbandingan dari kedua gerbang."""
    if standard_error is None or ci_low is None or ci_high is None:
        return INCONCLUSIVE
    if margin > NOISE_MULTIPLIER * standard_error and ci_low > 0:
        return SUPPORTED
    if margin < -NOISE_MULTIPLIER * standard_error and ci_high < 0:
        return OPPOSED
    return INCONCLUSIVE


def _spread(values: list[float]) -> tuple[float, float | None]:
    return statistics.mean(values), statistics.stdev(values) if len(values) >= 2 else None


def _ranked(arm: ArmScores, rows: EvaluationRows, mask: np.ndarray) -> list[RankedScores]:
    return [
        rank_scores(score[mask], rows.label[mask], rows.day_index[mask]) for score in arm.scores
    ]


def _seed_mean(ranked: list[RankedScores], day_weights: np.ndarray) -> float:
    return float(np.mean([weighted_average_precision(item, day_weights) for item in ranked]))


def _day_resamples(n_days: int) -> np.ndarray:
    """Bobot hari untuk setiap resampel, sama untuk semua lengan dan perbandingan."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return np.stack(
        [
            np.bincount(rng.integers(0, n_days, n_days), minlength=n_days).astype(float)
            for _ in range(BOOTSTRAP_RESAMPLES)
        ]
    )


def compare(
    spec: ComparisonSpec,
    segment: str,
    challenger: ArmScores,
    baseline: ArmScores,
    rows: EvaluationRows,
    mask: np.ndarray,
    resamples: np.ndarray,
) -> ComparisonResult:
    """Menghitung kedua gerbang satu perbandingan pada baris `mask`."""
    unit = np.ones(rows.n_days)
    challenger_ranked = _ranked(challenger, rows, mask)
    baseline_ranked = _ranked(baseline, rows, mask)
    challenger_mean, challenger_std = _spread(
        [weighted_average_precision(item, unit) for item in challenger_ranked]
    )
    baseline_mean, baseline_std = _spread(
        [weighted_average_precision(item, unit) for item in baseline_ranked]
    )
    margin = challenger_mean - baseline_mean
    standard_error = None
    if challenger_std is not None and baseline_std is not None:
        standard_error = (
            challenger_std**2 / len(challenger.seeds) + baseline_std**2 / len(baseline.seeds)
        ) ** 0.5

    fraud_days = rows.day_index[mask & (rows.label == 1)]
    differences = [
        _seed_mean(challenger_ranked, weights) - _seed_mean(baseline_ranked, weights)
        for weights in resamples
        # Resampel tanpa satu pun fraud membuat PR-AUC tidak terdefinisi, jadi dilewati.
        if weights[fraud_days].sum() > 0
    ]
    ci_low = ci_high = None
    if differences:
        tail = (1 - CONFIDENCE_LEVEL) / 2 * 100
        ci_low, ci_high = (float(value) for value in np.percentile(differences, [tail, 100 - tail]))
    return ComparisonResult(
        name=spec.name,
        role=spec.role,
        segment=segment,
        challenger=spec.challenger,
        baseline=spec.baseline,
        n_rows=int(mask.sum()),
        n_fraud=int(rows.label[mask].sum()),
        challenger_mean=challenger_mean,
        challenger_std=challenger_std,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        margin=margin,
        standard_error=standard_error,
        ci_low=ci_low,
        ci_high=ci_high,
        valid_resamples=len(differences),
        verdict=verdict(margin, standard_error, ci_low, ci_high),
    )


def arm_segment_summary(
    arm: ArmScores, rows: EvaluationRows, mask: np.ndarray, resamples: np.ndarray
) -> dict[str, float | None]:
    """PR-AUC rata-rata antar seed satu lengan pada satu segmen, dengan selang bootstrap."""
    ranked = _ranked(arm, rows, mask)
    mean = _seed_mean(ranked, np.ones(rows.n_days))
    fraud_days = rows.day_index[mask & (rows.label == 1)]
    values = [_seed_mean(ranked, weights) for weights in resamples if weights[fraud_days].sum() > 0]
    if not values:
        return {"mean": mean, "ci_low": None, "ci_high": None}
    tail = (1 - CONFIDENCE_LEVEL) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return {"mean": mean, "ci_low": float(low), "ci_high": float(high)}


def evaluation_rows(data: SequenceData, split: str) -> EvaluationRows:
    """Baris evaluasi menurut split temporal, dengan segmen yang sama persis dengan tier 1."""
    positions = data.split_rows[split].cpu().numpy()
    days = data.transaction_dt.cpu().numpy()[positions] // SECONDS_PER_DAY
    unique_days, day_index = np.unique(days, return_inverse=True)
    # Ukuran entitas adalah total transaksi per kunci entitas sepanjang data, hanya untuk laporan.
    entity_totals = pd.Series(data.card1).map(pd.Series(data.card1).value_counts()).to_numpy()
    txn_count_so_far = positions - data.entity_start.cpu().numpy()[positions]
    return EvaluationRows(
        transaction_id=data.transaction_id[positions],
        label=data.label.cpu().numpy()[positions].astype(float),
        day_index=day_index,
        n_days=len(unique_days),
        size_key=size_segment_keys(entity_totals[positions]),
        position_key=position_segment_keys(txn_count_so_far),
    )


def _latest_by_seed(runs: list[Run]) -> dict[int, Run]:
    latest: dict[int, Run] = {}
    for run in sorted(runs, key=lambda run: run.info.start_time, reverse=True):
        latest.setdefault(int(run.data.tags["seed"]), run)
    return latest


def tier1_scores(
    experiment_id: str, arm: str, rows: EvaluationRows, full_split: str | None
) -> ArmScores:
    """Skor pengulangan pembanding tier 1 untuk baris evaluasi.

    Args:
        experiment_id: Eksperimen pengujian.
        arm: `t1` atau `t1_plus`.
        rows: Baris evaluasi.
        full_split: Nama split kalau baris evaluasi harus sama persis dengan split tier 1;
            None untuk mode smoke, yang datanya dipotong berbeda.
    """
    runs = mlflow.MlflowClient().search_runs(
        [experiment_id],
        filter_string=(
            f"tags.arm = '{arm}' and tags.run_role = 'trial' and attributes.status = 'FINISHED'"
        ),
        max_results=1000,
    )
    by_seed = _latest_by_seed(runs)
    prepared = prepare_datasets(with_window_features=TIER1_ARMS[arm])
    if full_split is not None:
        expected = set(getattr(prepared, full_split).entity_key["TransactionID"])
        if expected != set(rows.transaction_id):
            raise RuntimeError(f"Baris split {full_split} tier 1 dan tier 2 tidak sama.")
    parts = (prepared.train, prepared.validation, prepared.test)
    features = pd.concat([part.features for part in parts])
    features.index = pd.concat([part.entity_key["TransactionID"] for part in parts]).to_numpy()
    matrix = features.loc[rows.transaction_id]

    seeds = sorted(by_seed)
    scores = []
    for seed in seeds:
        model = mlflow_lightgbm.load_model(logged_model_uri(by_seed[seed]))
        if not isinstance(model, LGBMClassifier):
            raise TypeError(f"Run {by_seed[seed].info.run_id} tidak mencatat LGBMClassifier.")
        scores.append(np.asarray(model.predict_proba(matrix))[:, 1])
    return ArmScores(
        arm=arm,
        seeds=seeds,
        run_ids=[by_seed[seed].info.run_id for seed in seeds],
        scores=np.stack(scores) if scores else np.empty((0, len(rows.label))),
    )


def tier2_scores(
    arm: str, runs: dict[int, Run], data: SequenceData, split: str, seq_len: int
) -> ArmScores:
    """Logit pengulangan satu lengan tier 2 untuk baris evaluasi."""
    mode = arm.removeprefix("t2_")
    seeds = sorted(runs)
    scores = []
    for seed in seeds:
        run_id = runs[seed].info.run_id
        _, config, spec, state = load_run_package(run_id, FINETUNE_CHECKPOINT_ARTIFACT)
        ensure_same_spec(data, spec, run_id)
        model = Tier2Model(data.spec, config).to(data.numeric.device)
        model.load_state_dict(state)
        logits, _ = predict(model, data, split, arm_seq_len(mode, seq_len), mode)
        scores.append(logits)
    return ArmScores(
        arm=arm,
        seeds=seeds,
        run_ids=[runs[seed].info.run_id for seed in seeds],
        scores=np.stack(scores) if scores else np.empty((0, int(len(data.split_rows[split])))),
    )


def completed_test_evaluations(experiment_id: str) -> list[str]:
    """Run evaluasi split uji yang sudah selesai, dasar penjaga satu kali."""
    runs = mlflow.MlflowClient().search_runs(
        [experiment_id],
        filter_string=(
            f"tags.stage = '{STAGE}' and tags.split = 'test' and attributes.status = 'FINISHED'"
        ),
    )
    return [run.info.run_id for run in runs]


def _plot_pr_curves(arms: dict[str, ArmScores], rows: EvaluationRows, path: Path) -> None:
    frames = []
    for arm in ARM_ORDER:
        if arm not in arms or not len(arms[arm].seeds):
            continue
        precision, recall, _ = precision_recall_curve(rows.label, arms[arm].scores[0])
        keep = np.unique(np.linspace(0, len(recall) - 1, PR_CURVE_POINTS).astype(int))
        frames.append(
            pd.DataFrame({"recall": recall[keep], "precision": precision[keep], "arm": arm})
        )
    figure = Figure(figsize=(7, 5))
    axis = figure.subplots()
    # Kurva PR digambar dari titik mentah; agregasi bawaan seaborn akan merata-ratakan titik
    # dengan recall sama dan menambahkan pita selang yang tidak bermakna di sini.
    sns.lineplot(
        data=pd.concat(frames), x="recall", y="precision", hue="arm",
        palette=ARM_COLORS, linewidth=2, estimator=None, sort=False, ax=axis,
    )
    axis.set(xlabel="Recall", ylabel="Precision", title="Kurva PR per lengan (seed pertama)")
    # Di luar area plot supaya tidak menutupi kurva dan titik data.
    axis.legend(title="Lengan", loc="upper left", bbox_to_anchor=(1.01, 1))
    figure.savefig(path, dpi=150, bbox_inches="tight")


def _plot_seed_spread(arm_values: dict[str, list[float]], path: Path) -> None:
    frame = pd.DataFrame(
        [(arm, value) for arm in ARM_ORDER for value in arm_values.get(arm, [])],
        columns=["arm", "pr_auc"],
    )
    figure = Figure(figsize=(7, 4))
    axis = figure.subplots()
    sns.stripplot(
        data=frame, x="arm", y="pr_auc", hue="arm", palette=ARM_COLORS, size=8, legend=False,
        order=[arm for arm in ARM_ORDER if arm in arm_values], ax=axis,
    )
    axis.set(xlabel="Lengan", ylabel="PR-AUC", title="PR-AUC tiap seed per lengan")
    figure.savefig(path, dpi=150, bbox_inches="tight")


def _plot_size_segments(summary: dict[str, dict[str, dict]], path: Path) -> None:
    segments = [key for key, _ in SIZE_SEGMENTS]
    arms = [arm for arm in ARM_ORDER if arm in summary]
    figure = Figure(figsize=(8, 5))
    axis = figure.subplots()
    width = 0.7 / max(len(arms), 1)
    for offset, arm in enumerate(arms):
        for position, segment in enumerate(segments):
            values = summary[arm][segment]
            x = position - 0.35 + width * (offset + 0.5)
            low, high = values["ci_low"], values["ci_high"]
            error = None if low is None else [[values["mean"] - low], [high - values["mean"]]]
            axis.errorbar(
                x, values["mean"], yerr=error, fmt="o", markersize=8, linewidth=2, capsize=3,
                color=ARM_COLORS[arm], label=arm if position == 0 else None,
            )
    axis.set_xticks(range(len(segments)), [label for _, label in SIZE_SEGMENTS], fontsize=8)
    axis.set(ylabel="PR-AUC", title="PR-AUC per segmen ukuran entitas (selang bootstrap 95%)")
    # Di luar area plot supaya tidak menutupi kurva dan titik data.
    axis.legend(title="Lengan", loc="upper left", bbox_to_anchor=(1.01, 1))
    figure.savefig(path, dpi=150, bbox_inches="tight")


def _print_results(arm_values: dict[str, list[float]], results: list[ComparisonResult]) -> None:
    print("\nPR-AUC per lengan (rata-rata +/- simpangan baku antar seed):")
    for arm in ARM_ORDER:
        values = arm_values.get(arm, [])
        if values:
            mean, std = _spread(values)
            spread = f"+/- {std:.4f}" if std is not None else "(1 seed)"
            print(f"  {arm:16s} {mean:.4f} {spread} ({len(values)} seed)")
    print("\nPerbandingan:")
    for result in results:
        interval = (
            f"[{result.ci_low:+.4f}, {result.ci_high:+.4f}]"
            if result.ci_low is not None and result.ci_high is not None
            else "[-]"
        )
        threshold = (
            f"{NOISE_MULTIPLIER * result.standard_error:.4f}"
            if result.standard_error is not None
            else "-"
        )
        print(
            f"  {result.name} [{result.segment}] ({result.role}): selisih {result.margin:+.4f}, "
            f"batas gerbang 1 {threshold}, selang 95% {interval} -> {result.verdict}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluasi pengujian hipotesis tier 2.")
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--parquet-dir", type=Path, default=INITIAL_PARQUET_DIR)
    parser.add_argument("--smoke", action="store_true", help="Eksperimen smoke.")
    args = parser.parse_args()

    check_average_precision()
    experiment_id = setup_mlflow(args.smoke)
    if args.split == "test" and completed_test_evaluations(experiment_id):
        raise SystemExit(
            "Evaluasi split uji sudah pernah selesai untuk paket lengan ini; split uji hanya "
            f"boleh dipakai sekali (run {completed_test_evaluations(experiment_id)})."
        )
    selected = selected_configuration(experiment_id, args.smoke)
    if selected is None:
        raise SystemExit("Pemilihan konfigurasi tier 2 belum lengkap.")
    seq_len, config = selected
    required = REQUIRED_SEEDS[:SMOKE_SEED_COUNT] if args.smoke else REQUIRED_SEEDS

    main_runs = latest_main_runs(experiment_id, seq_len, config)
    tier2_runs = {
        arm: {seed: run for (mode, seed), run in main_runs.items() if f"t2_{mode}" == arm}
        for arm in TIER2_ARMS
    }
    if args.split == "test":
        missing = [
            f"{arm} seed {seed}"
            for arm in TIER2_ARMS
            for seed in required
            if seed not in tier2_runs[arm]
        ]
        if missing:
            raise SystemExit("Pengulangan tier 2 belum lengkap: " + ", ".join(missing))

    device = select_device()
    data = prepare_sequence_data(args.parquet_dir, SMOKE_ROW_LIMIT if args.smoke else None)
    data = data.to(device)
    rows = evaluation_rows(data, args.split)
    full_split = None if args.smoke else args.split

    arms: dict[str, ArmScores] = {}
    for arm in TIER1_ARMS:
        arms[arm] = tier1_scores(experiment_id, arm, rows, full_split)
    for arm in TIER2_ARMS:
        if tier2_runs[arm]:
            arms[arm] = tier2_scores(arm, tier2_runs[arm], data, args.split, seq_len)
    if args.split == "test":
        incomplete = [arm for arm in ARM_ORDER if set(arms[arm].seeds) < set(required)]
        if incomplete:
            raise SystemExit("Pengulangan belum lengkap untuk lengan: " + ", ".join(incomplete))

    resamples = _day_resamples(rows.n_days)
    everything = np.ones(len(rows.label), dtype=bool)
    specs = ([PROMOTION] if args.split == "validation" else []) + [
        ComparisonSpec(spec.name, spec.challenger, spec.baseline, "diagnostic", spec.by_size)
        if args.split == "validation"
        else spec
        for spec in HYPOTHESIS_COMPARISONS
    ]
    results = []
    for spec in specs:
        if not (spec.challenger in arms and spec.baseline in arms):
            continue
        segments = [("global", everything)]
        if spec.by_size:
            segments += [(key, rows.size_key == key) for key, _ in SIZE_SEGMENTS]
        for segment, mask in segments:
            print(f"Menghitung {spec.name} [{segment}] ...", flush=True)
            results.append(
                compare(
                    spec, segment, arms[spec.challenger], arms[spec.baseline], rows, mask,
                    resamples,
                )
            )

    unit = np.ones(rows.n_days)
    arm_values = {
        arm: [
            weighted_average_precision(item, unit) for item in _ranked(scores, rows, everything)
        ]
        for arm, scores in arms.items()
        if len(scores.seeds)
    }
    size_summary = {
        arm: {
            key: arm_segment_summary(arms[arm], rows, rows.size_key == key, resamples)
            for key, _ in SIZE_SEGMENTS
        }
        for arm in arm_values
    }
    position_summary = {
        arm: {
            key: _seed_mean(_ranked(arms[arm], rows, rows.position_key == key), unit)
            if (rows.label[rows.position_key == key] == 1).any()
            else None
            for key, _ in POSITION_SEGMENTS
        }
        for arm in arm_values
    }
    _print_results(arm_values, results)

    reference = None
    if args.split == "test":
        reference = mlflow.MlflowClient().get_run(REFERENCE_TIER1_RUN_ID).data.metrics
    report = {
        "split": args.split,
        "criteria": {
            "noise_multiplier": NOISE_MULTIPLIER,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_unit": "hari",
            "confidence_level": CONFIDENCE_LEVEL,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "required_seeds": list(required),
        },
        "selected_configuration": {"seq_len": seq_len, **config.to_dict()},
        "rows": {"n_rows": int(len(rows.label)), "n_fraud": int(rows.label.sum())},
        "arms": {
            arm: {"seeds": scores.seeds, "run_ids": scores.run_ids, "pr_auc": arm_values[arm]}
            for arm, scores in arms.items()
            if arm in arm_values
        },
        "comparisons": [asdict(result) for result in results],
        "size_segments": size_summary,
        "position_segments": position_summary,
        "reference_tier1_v1_test": reference,
    }

    with mlflow.start_run(run_name=f"{STAGE}-{args.split}"):
        mlflow.set_tags({"stage": STAGE, "split": args.split})
        mlflow.log_params(
            {
                "split": args.split,
                "noise_multiplier": NOISE_MULTIPLIER,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "confidence_level": CONFIDENCE_LEVEL,
                "selected_seq_len": seq_len,
                "selected_d_model": config.d_model,
                "selected_n_layers": config.n_layers,
            }
        )
        metrics = {}
        for arm, values in arm_values.items():
            metrics[f"{arm}_pr_auc_mean"] = statistics.mean(values)
        for result in results:
            prefix = f"{result.name}_{result.segment}"
            metrics[f"{prefix}_margin"] = result.margin
            metrics[f"{prefix}_verdict_code"] = VERDICT_CODES[result.verdict]
        mlflow.log_metrics(metrics)
        mlflow.log_dict(report, "evaluation_report.json")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            sns.set_theme(style="whitegrid", rc={"figure.facecolor": SURFACE_COLOR})
            _plot_pr_curves(arms, rows, folder / "pr_curves.png")
            _plot_seed_spread(arm_values, folder / "seed_spread.png")
            _plot_size_segments(size_summary, folder / "size_segments.png")
            mlflow.log_artifacts(str(folder), "figures")

    promotion = next((result for result in results if result.role == "promotion"), None)
    if promotion is not None:
        recommendation = (
            "pindahkan alias production ke T1+"
            if promotion.verdict == SUPPORTED
            else "pertahankan tier 1 versi 1 sebagai production"
        )
        print(f"\nRekomendasi promosi (keputusan tetap di tangan manusia): {recommendation}")
    print(json.dumps(report["rows"]), flush=True)


if __name__ == "__main__":
    main()
