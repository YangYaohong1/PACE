from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .collector import collect_dataset, default_processed_dir


@dataclass(frozen=True)
class RunConfig:
    temperature: float
    iterations: int
    repetitions: int
    mode: str
    particles: int
    cost: float
    dataset: str
    interaction: str
    curve: str
    wandb_mode: str | None = None
    task_id: int | None = None
    cpus_per_task: int | None = None


def build_command(repo_root: Path, config: RunConfig) -> list[str]:
    script_path = repo_root / "dp_exp.py"
    return [
        sys.executable,
        str(script_path),
        str(config.temperature),
        str(config.iterations),
        str(config.repetitions),
        config.mode,
        str(config.particles),
        str(config.cost),
        config.dataset,
        config.interaction,
        config.curve,
    ]


def _write_run_manifest(
    repo_root: Path,
    config: RunConfig,
    processed_csv: Path,
    command: list[str],
    dry_run: bool,
) -> Path:
    runs_dir = repo_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runs_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{processed_csv.stem}.json"
    payload = {
        "created_at": datetime.now().isoformat(),
        "processed_dataset": str(processed_csv),
        "command": command,
        "working_directory": str(repo_root),
        "dry_run": dry_run,
        "config": asdict(config),
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def run_experiment(
    repo_root: Path,
    config: RunConfig,
    processed_dir: Path | None = None,
    dry_run: bool = False,
    skip_collect: bool = False,
) -> dict[str, object]:
    processed_dir = processed_dir or default_processed_dir(repo_root)
    collected = None if skip_collect else collect_dataset(config.dataset, repo_root=repo_root, output_dir=processed_dir)
    if skip_collect:
        processed_csv = processed_dir / f"{config.dataset.replace('/', '__')}.csv"
        if not processed_csv.exists():
            raise FileNotFoundError(
                f"Expected processed dataset at {processed_csv}. "
                "Run PACE/collect_data.py first or omit --skip-collect."
            )
    else:
        processed_csv = Path(collected["csv_path"])

    command = build_command(repo_root, config)
    env = os.environ.copy()
    if config.wandb_mode:
        env["WANDB_MODE"] = config.wandb_mode
    if config.task_id is not None:
        env["SLURM_ARRAY_TASK_ID"] = str(config.task_id)
    if config.cpus_per_task is not None:
        env["SLURM_CPUS_PER_TASK"] = str(config.cpus_per_task)

    result = {
        "command": command,
        "processed_csv": processed_csv,
        "cwd": repo_root,
        "dry_run": dry_run,
    }

    manifest_path = _write_run_manifest(repo_root, config, processed_csv, command, dry_run)
    result["manifest_path"] = manifest_path

    if dry_run:
        return result

    subprocess.run(command, cwd=repo_root, env=env, check=True)
    return result


# Backwards-compatible name for older local commands.
run_original_experiment = run_experiment
