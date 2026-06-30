from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata for one dataset that can be used by the PACE experiment."""

    name: str
    source_file: str
    epsilon_column: str
    accuracy_column: str
    kind: str
    description: str
    dataset_filter: str | None = None

    @property
    def slug(self) -> str:
        return self.name.replace("/", "__")

    def source_path(self, repo_root: Path) -> Path:
        return repo_root / self.source_file


DATASET_SPECS: dict[str, DatasetSpec] = {
    "adult": DatasetSpec(
        name="adult",
        source_file="data/raw/optimal_results_adult_N_TRIALS_20.csv",
        epsilon_column="epsilon",
        accuracy_column="best_test_accuracy",
        kind="logistic",
        description="Adult dataset with tuned DP logistic regression results.",
    ),
    "dutch": DatasetSpec(
        name="dutch",
        source_file="data/raw/optimal_results_dutch_N_TRIALS_20.csv",
        epsilon_column="epsilon",
        accuracy_column="best_test_accuracy",
        kind="logistic",
        description="Dutch dataset with tuned DP logistic regression results.",
    ),
    "cifar100": DatasetSpec(
        name="cifar100",
        source_file="data/raw/epsilon-accuracy.csv",
        epsilon_column="target_epsilon",
        accuracy_column="accuracy",
        kind="benchmark",
        description="CIFAR100 benchmark results from the shared epsilon/accuracy table.",
        dataset_filter="cifar100",
    ),
    "dpdl-benchmark/patch_camelyon": DatasetSpec(
        name="dpdl-benchmark/patch_camelyon",
        source_file="data/raw/epsilon-accuracy.csv",
        epsilon_column="target_epsilon",
        accuracy_column="accuracy",
        kind="benchmark",
        description="Patch Camelyon benchmark results from the shared epsilon/accuracy table.",
        dataset_filter="dpdl-benchmark/patch_camelyon",
    ),
    "dpdl-benchmark/sun397": DatasetSpec(
        name="dpdl-benchmark/sun397",
        source_file="data/raw/epsilon-accuracy.csv",
        epsilon_column="target_epsilon",
        accuracy_column="accuracy",
        kind="benchmark",
        description="SUN397 benchmark results from the shared epsilon/accuracy table.",
        dataset_filter="dpdl-benchmark/sun397",
    ),
    "dpdl-benchmark/svhn_cropped": DatasetSpec(
        name="dpdl-benchmark/svhn_cropped",
        source_file="data/raw/epsilon-accuracy.csv",
        epsilon_column="target_epsilon",
        accuracy_column="accuracy",
        kind="benchmark",
        description="SVHN benchmark results from the shared epsilon/accuracy table.",
        dataset_filter="dpdl-benchmark/svhn_cropped",
    ),
}


def available_dataset_names() -> list[str]:
    return sorted(DATASET_SPECS)


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[name]
    except KeyError as exc:
        available = ", ".join(available_dataset_names())
        raise KeyError(f"Unknown dataset {name!r}. Available datasets: {available}") from exc
