from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import DatasetSpec, available_dataset_names, get_dataset_spec


def repo_root_from_file(file_path: str) -> Path:
    return Path(file_path).resolve().parent


def default_processed_dir(repo_root: Path) -> Path:
    return repo_root / "data" / "processed"


def _normalize_log_epsilons(epsilons: pd.Series) -> np.ndarray:
    log_epsilons = np.log(epsilons.to_numpy(dtype=float))
    min_value = float(log_epsilons.min())
    max_value = float(log_epsilons.max())
    if np.isclose(min_value, max_value):
        return np.ones_like(log_epsilons)
    scaled = (log_epsilons - min_value) / (max_value - min_value)
    return 1.0 - scaled


def _normalize_accuracy(accuracy: pd.Series) -> np.ndarray:
    values = accuracy.to_numpy(dtype=float)
    min_value = float(values.min())
    max_value = float(values.max())
    if np.isclose(min_value, max_value):
        return np.ones_like(values)
    return (values - min_value) / (max_value - min_value)


def load_dataset_frame(spec: DatasetSpec, repo_root: Path) -> pd.DataFrame:
    source_path = spec.source_path(repo_root)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source data for {spec.name}: {source_path}")

    frame = pd.read_csv(source_path)
    if spec.dataset_filter is not None:
        frame = frame.loc[frame["dataset_name"] == spec.dataset_filter].copy()

    if frame.empty:
        raise ValueError(f"No rows found for dataset {spec.name!r} in {source_path}")

    standardized = frame.rename(
        columns={
            spec.epsilon_column: "epsilon",
            spec.accuracy_column: "accuracy",
        }
    ).copy()
    standardized = standardized.sort_values("epsilon").reset_index(drop=True)
    standardized["dataset_name"] = spec.name
    standardized["privacy_score"] = _normalize_log_epsilons(standardized["epsilon"])
    standardized["normalized_accuracy"] = _normalize_accuracy(standardized["accuracy"])
    standardized["source_file"] = str(source_path.relative_to(repo_root))
    standardized["source_kind"] = spec.kind
    standardized["row_id"] = np.arange(len(standardized))

    columns = [
        "dataset_name",
        "row_id",
        "epsilon",
        "accuracy",
        "privacy_score",
        "normalized_accuracy",
        "source_kind",
        "source_file",
    ]
    return standardized.loc[:, columns]


def summarize_dataset(frame: pd.DataFrame) -> dict[str, float | int | str]:
    return {
        "dataset_name": str(frame["dataset_name"].iloc[0]),
        "num_rows": int(len(frame)),
        "epsilon_min": float(frame["epsilon"].min()),
        "epsilon_max": float(frame["epsilon"].max()),
        "accuracy_min": float(frame["accuracy"].min()),
        "accuracy_max": float(frame["accuracy"].max()),
    }


def collect_dataset(
    dataset_name: str,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, object]:
    spec = get_dataset_spec(dataset_name)
    output_dir = output_dir or default_processed_dir(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_dataset_frame(spec, repo_root)
    summary = summarize_dataset(frame)

    csv_path = output_dir / f"{spec.slug}.csv"
    json_path = output_dir / f"{spec.slug}.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return {
        "dataset": spec.name,
        "csv_path": csv_path,
        "summary_path": json_path,
        "summary": summary,
    }


def collect_many(
    dataset_names: list[str],
    repo_root: Path,
    output_dir: Path | None = None,
) -> list[dict[str, object]]:
    return [collect_dataset(name, repo_root=repo_root, output_dir=output_dir) for name in dataset_names]


def collect_all(repo_root: Path, output_dir: Path | None = None) -> list[dict[str, object]]:
    return collect_many(available_dataset_names(), repo_root=repo_root, output_dir=output_dir)
