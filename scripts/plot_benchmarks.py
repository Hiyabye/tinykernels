#!/usr/bin/env python3
"""
Plot tinykernels benchmark CSV results.

Expected CSV columns:
  sweep,rows,inner,cols,backend,loop_order,use_blocking,num_threads,
  block_size,iterations,time_sec,speedup_vs_baseline,label

Usage:
  python3 scripts/plot_benchmarks.py benchmark_results.csv assets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LABEL_ORDER = [
    "single_plain_ijk",
    "single_plain_ikj",
    "single_blocked_ijk",
    "single_blocked_ikj",
    "pthread_plain_ijk",
    "pthread_plain_ikj",
    "pthread_blocked_ijk",
    "pthread_blocked_ikj",
    "openmp_plain_ijk",
    "openmp_plain_ikj",
    "openmp_blocked_ijk",
    "openmp_blocked_ikj",
]


def _ordered_labels(data: pd.DataFrame) -> list[str]:
    available = set(data["label"].unique())
    ordered = [label for label in LABEL_ORDER if label in available]
    remaining = sorted(available - set(ordered))
    return ordered + remaining


def _finish_plot(output_path: Path, *, log_y: bool = False) -> None:
    if log_y:
        plt.yscale("log")
    plt.grid(True, axis="y", alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_matrix_size_sweep(df: pd.DataFrame, output_dir: Path) -> None:
    data = df[df["sweep"] == "matrix_size"].copy()
    if data.empty:
        return

    plt.figure(figsize=(10.0, 6.0))
    for label in _ordered_labels(data):
        subset = data[data["label"] == label].sort_values("rows")
        if subset.empty:
            continue
        plt.plot(subset["rows"], subset["time_sec"], marker="o", label=label)

    plt.xlabel("Matrix size N for NxN × NxN")
    plt.ylabel("Median time (seconds, log scale)")
    plt.title("Matrix size sweep")
    _finish_plot(output_dir / "matrix_size_sweep.png", log_y=True)


def plot_thread_count_sweep(df: pd.DataFrame, output_dir: Path) -> None:
    data = df[df["sweep"] == "thread_count"].copy()
    if data.empty:
        return

    plt.figure(figsize=(10.0, 6.0))
    for label in _ordered_labels(data):
        subset = data[data["label"] == label].sort_values("num_threads")
        if subset.empty:
            continue
        plt.plot(
            subset["num_threads"],
            subset["time_sec"],
            marker="o",
            label=label,
        )

    plt.xlabel("Threads")
    plt.ylabel("Median time (seconds)")
    plt.title("Thread count sweep")
    _finish_plot(output_dir / "thread_count_sweep.png")


def plot_block_size_sweep(df: pd.DataFrame, output_dir: Path) -> None:
    data = df[df["sweep"] == "block_size"].copy()
    if data.empty:
        return

    plt.figure(figsize=(10.0, 6.0))
    for label in _ordered_labels(data):
        subset = data[data["label"] == label].sort_values("block_size")
        if subset.empty:
            continue
        plt.plot(subset["block_size"], subset["time_sec"], marker="o", label=label)

    plt.xlabel("Block size")
    plt.ylabel("Median time (seconds)")
    plt.title("Block size sweep")
    _finish_plot(output_dir / "block_size_sweep.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv_path)

    plot_matrix_size_sweep(df, args.output_dir)
    plot_thread_count_sweep(df, args.output_dir)
    plot_block_size_sweep(df, args.output_dir)


if __name__ == "__main__":
    main()
