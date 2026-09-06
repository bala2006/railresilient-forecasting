"""Command-line interface for the RailResilient-JP pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from railresilient.config import load_config
from railresilient.data import (
    download_ride_months,
    estimate_download_gib,
    prepare_datasets,
)
from railresilient.experiment import run_experiments
from railresilient.odpt import check_access, collect_alert_snapshot
from railresilient.reporting import write_reports


def _common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", default="configs/pilot.json", help="Path to JSON configuration"
    )


def _common_roots(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default="data", help="Dataset root")
    parser.add_argument("--artifact-root", default="artifacts", help="Artifact root")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="railresilient",
        description="CPU-first railway delay forecasting under unreliable feeds",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate = subparsers.add_parser("estimate", help="Estimate selected RIDE download size")
    _common_config(estimate)

    download = subparsers.add_parser("download", help="Download pinned RIDE Silver months")
    _common_config(download)
    download.add_argument("--data-root", default="data", help="Dataset root")

    prepare = subparsers.add_parser("prepare", help="Build leakage-safe chronological datasets")
    _common_config(prepare)
    prepare.add_argument("--data-root", default="data", help="Dataset root")

    run = subparsers.add_parser("run", help="Train and evaluate all pilot models")
    _common_config(run)
    _common_roots(run)
    run.add_argument("--run-name", help="Explicit artifact run directory name")

    report = subparsers.add_parser("report", help="Generate cards, CSV, and findings")
    report.add_argument("--run-dir", required=True, help="Completed run directory")
    report.add_argument(
        "--dataset-manifest",
        default="data/manifests/prepared_dataset_manifest.json",
        help="Prepared dataset manifest",
    )

    run_all = subparsers.add_parser(
        "run-all", help="Download, prepare, train, evaluate, and report"
    )
    _common_config(run_all)
    _common_roots(run_all)
    run_all.add_argument("--run-name", help="Explicit artifact run directory name")
    run_all.add_argument(
        "--skip-download", action="store_true", help="Use existing raw files"
    )
    run_all.add_argument(
        "--skip-prepare", action="store_true", help="Use existing prepared arrays"
    )

    subparsers.add_parser("odpt-check", help="Check token-gated ODPT resources")
    snapshot = subparsers.add_parser(
        "odpt-snapshot", help="Collect one token-gated Tokyo Metro alert snapshot"
    )
    snapshot.add_argument("--output-root", default="data/private/odpt")
    return parser


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "estimate":
            config = load_config(args.config)
            _print_json({"estimated_download_gib": estimate_download_gib(config)})
        elif args.command == "download":
            _print_json(download_ride_months(load_config(args.config), args.data_root))
        elif args.command == "prepare":
            _print_json(prepare_datasets(load_config(args.config), args.data_root))
        elif args.command == "run":
            run_dir = run_experiments(
                load_config(args.config),
                args.data_root,
                args.artifact_root,
                args.run_name,
            )
            _print_json({"run_dir": str(run_dir.resolve())})
        elif args.command == "report":
            write_reports(args.run_dir, args.dataset_manifest)
            _print_json({"reported_run_dir": str(Path(args.run_dir).resolve())})
        elif args.command == "run-all":
            config = load_config(args.config)
            if not args.skip_download:
                download_ride_months(config, args.data_root)
            if not args.skip_prepare:
                prepare_datasets(config, args.data_root)
            run_dir = run_experiments(
                config, args.data_root, args.artifact_root, args.run_name
            )
            manifest = Path(args.data_root) / "manifests" / "prepared_dataset_manifest.json"
            write_reports(run_dir, manifest)
            _print_json({"run_dir": str(run_dir.resolve()), "reports": "complete"})
        elif args.command == "odpt-check":
            _print_json(check_access())
        elif args.command == "odpt-snapshot":
            output = collect_alert_snapshot(args.output_root)
            _print_json({"snapshot": str(output.resolve())})
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
    except Exception as error:  # CLI boundary: emit concise diagnostics and non-zero status.
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
