from __future__ import annotations

import argparse
from pathlib import Path

from pace.catalog import available_dataset_names
from pace.collector import collect_all, collect_dataset, default_processed_dir, repo_root_from_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect and normalize the datasets bundled with PACE.",
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help="Dataset to collect. Use 'all' to export every supported dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where normalized CSV and summary JSON files will be written.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the supported dataset names and exit.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    repo_root = repo_root_from_file(__file__)
    output_dir = args.output_dir or default_processed_dir(repo_root)

    if args.list:
        for name in available_dataset_names():
            print(name)
        return

    if args.dataset == "all":
        results = collect_all(repo_root=repo_root, output_dir=output_dir)
    else:
        results = [collect_dataset(args.dataset, repo_root=repo_root, output_dir=output_dir)]

    for item in results:
        summary = item["summary"]
        print(
            f"{item['dataset']}: wrote {item['csv_path']} "
            f"({summary['num_rows']} rows, epsilon {summary['epsilon_min']:.5f}-{summary['epsilon_max']:.5f})"
        )


if __name__ == "__main__":
    main()
