from __future__ import annotations

import argparse
from pathlib import Path

from pace.catalog import available_dataset_names
from pace.collector import default_processed_dir, repo_root_from_file
from pace.runner import RunConfig, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the standalone PACE DP experiment with dataset validation.",
    )
    parser.add_argument("--temperature", type=float, required=True, help="Boltzmann temperature T.")
    parser.add_argument("--iterations", type=int, required=True, help="Number of experiment iterations.")
    parser.add_argument("--repetitions", type=int, required=True, help="Number of repeated runs.")
    parser.add_argument(
        "--mode",
        choices=["sequential", "interleaved"],
        required=True,
        help="Experiment mode.",
    )
    parser.add_argument("--particles", type=int, required=True, help="Number of particles for curve inference.")
    parser.add_argument("--cost", type=float, required=True, help="Preference-query cost.")
    parser.add_argument("--dataset", choices=available_dataset_names(), required=True, help="Dataset to use.")
    parser.add_argument(
        "--interaction",
        choices=["true", "pair", "curve"],
        required=True,
        help="Interaction type.",
    )
    parser.add_argument(
        "--curve",
        choices=["sigmoid", "gompertz"],
        required=True,
        help="Curve family.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Directory containing normalized dataset exports from collect_data.py.",
    )
    parser.add_argument(
        "--task-id", type=int, default=0, help="Optional SLURM_ARRAY_TASK_ID override."
    )
    parser.add_argument("--cpus-per-task", type=int, default=None, help="Optional SLURM_CPUS_PER_TASK override.")
    parser.add_argument("--skip-collect", action="store_true", help="Skip dataset export before running.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths and command without executing.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = repo_root_from_file(__file__)
    processed_dir = args.processed_dir or default_processed_dir(repo_root)

    config = RunConfig(
        temperature=args.temperature,
        iterations=args.iterations,
        repetitions=args.repetitions,
        mode=args.mode,
        particles=args.particles,
        cost=args.cost,
        dataset=args.dataset,
        interaction=args.interaction,
        curve=args.curve,
        task_id=args.task_id,
        cpus_per_task=args.cpus_per_task,
    )

    result = run_experiment(
        repo_root=repo_root,
        config=config,
        processed_dir=processed_dir,
        dry_run=args.dry_run,
        skip_collect=args.skip_collect,
    )

    print(f"Processed dataset: {result['processed_csv']}")
    print(f"Run manifest: {result['manifest_path']}")
    print(f"Working directory: {result['cwd']}")
    print("Command:")
    print(" ".join(result["command"]))


if __name__ == "__main__":
    main()
