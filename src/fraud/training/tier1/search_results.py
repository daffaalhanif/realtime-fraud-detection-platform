"""Utilitas baca dari MLflow untuk hasil pelatihan tier 1, dipakai lintas beberapa modul.

Dipisah dari `search.py` supaya modul yang bukan bagian dari strategi pencarian hyperparameter
(misalnya kalibrasi) tidak perlu mengimpor dari modul strategi pencarian, hanya dari sini.
`search.py` sendiri tetap memakai fungsi-fungsi ini untuk kebutuhannya sendiri (memilih sumber
phase 2 dan konfirmasi).
"""

from dataclasses import dataclass
from typing import Any

import mlflow
import mlflow.artifacts
from mlflow.entities import Run

from fraud.training.tier1.candidates import MODEL_LOADERS, STAGE_CONFIRM


@dataclass(frozen=True)
class Confirmation:
    """Hasil confirm (pelatihan ulang beberapa seed) untuk satu run sumber.

    Attributes:
        mean: Rata-rata PR-AUC validasi lintas seed.
        std: Simpangan baku PR-AUC validasi lintas seed.
        n_seeds: Jumlah seed yang dipakai.
    """

    mean: float
    std: float
    n_seeds: int


def finished_parents(experiment_id: str, name: str) -> list[Run]:
    """Run induk kandidat yang sudah selesai, dari yang terbaru."""
    return mlflow.MlflowClient().search_runs(
        [experiment_id],
        filter_string=(
            f"tags.candidate = '{name}' and tags.run_role = 'parent' "
            "and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=200,
    )


def best_parent(experiment_id: str, name: str, stages: tuple[str, ...]) -> Run:
    """Run induk satu kandidat dengan PR-AUC validasi tertinggi di antara tahap-tahap itu.

    Dipilih berdasarkan skor, bukan yang terbaru, karena ronde pencarian yang lebih baru
    belum tentu lebih baik dari yang lama.
    """
    candidates = [
        run
        for run in finished_parents(experiment_id, name)
        if run.data.tags.get("search_stage") in stages
    ]
    if not candidates:
        raise RuntimeError(f"Belum ada run {'/'.join(stages)} yang selesai untuk {name}.")
    return max(candidates, key=lambda run: run.data.metrics["best_validation_pr_auc"])


def confirmation_of(experiment_id: str, name: str, source_run_id: str) -> Confirmation | None:
    """Hasil confirm terbaru untuk sebuah run sumber, atau None kalau belum dikonfirmasi.

    Dicocokkan lewat `source_run_id` yang dicatat run confirm, bukan lewat urutan waktu,
    karena confirm milik konfigurasi lain tidak boleh terbaca sebagai milik run ini.
    """
    runs = mlflow.MlflowClient().search_runs(
        [experiment_id],
        filter_string=(
            f"tags.candidate = '{name}' and tags.search_stage = '{STAGE_CONFIRM}' "
            f"and tags.run_role = 'parent' and params.source_run_id = '{source_run_id}' "
            "and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None
    run = runs[0]
    return Confirmation(
        mean=run.data.metrics["pr_auc_mean"],
        std=run.data.metrics["pr_auc_std"],
        n_seeds=len(run.data.params["seeds"].split(",")),
    )


def load_run_dict(run: Run, artifact: str) -> dict[str, Any]:
    """Membaca artefak JSON dari sebuah run."""
    return mlflow.artifacts.load_dict(f"runs:/{run.info.run_id}/{artifact}")


def logged_model_uri(run: Run) -> str:
    """URI model yang tercatat di sebuah run.

    Raises:
        RuntimeError: Kalau run ini tidak mencatat model apa pun.
    """
    logged = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        filter_string=f"source_run_id = '{run.info.run_id}'",
        output_format="list",
    )
    if not logged:
        raise RuntimeError(f"Run {run.info.run_id} tidak mencatat model apa pun.")
    return f"models:/{logged[0].model_id}"


def load_model(run: Run, name: str) -> Any:
    """Memuat model terlatih hasil `log_best` (run induk phase 1 atau phase 2).

    Hanya untuk model asli kandidat. Model hasil kalibrasi berformat sklearn apa pun kandidat
    asalnya, jadi dimuat lewat `logged_model_uri` dengan loader sklearn, bukan lewat sini.

    Args:
        run: Run induk phase 1 atau phase 2 yang mencatat model.
        name: Nama kandidat, kunci di `fraud.training.tier1.candidates.MODEL_LOADERS`.

    Returns:
        Model terlatih siap pakai (`predict_proba`, dan seterusnya).

    Raises:
        RuntimeError: Kalau run ini tidak mencatat model apa pun.
    """
    return MODEL_LOADERS[name](logged_model_uri(run))
