#!/usr/bin/env python3
"""
Plot tinykernels benchmark results.

Expected CSV columns:
  sweep,n,threads,block_size,iterations,kernel,time_sec,speedup_vs_ref

Usage:
  python3 scripts/plot_benchmarks.py benchmark_results.csv assets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

KERNEL_ORDER = [
    "Reference IJK",
    "Sequential IKJ",
    "Parallel Rows IJK",
    "Sequential Blocked IKJ",
    "Parallel Rows Blocked IKJ",
    "OpenMP IKJ",
]


def _finish_plot(output_path: Path) -> None:
    plt.grid(True, axis="y", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_matrix_size_sweep(df: pd.DataFrame, output_dir: Path) -> None:
    data = df[df["sweep"] == "matrix_size"].copy()

    plt.figure(figsize=(9, 5.2))
    for kernel in KERNEL_ORDER:
        subset = data[data["kernel"] == kernel].sort_values("n")
        if subset.empty:
            continue
        plt.plot(subset["n"], subset["time_sec"], marker="o", label=kernel)

    plt.yscale("log")
    plt.xlabel("Matrix size N for NxN × NxN")
    plt.ylabel("Median time (seconds, log scale)")
    plt.title("Matrix size sweep")
    _finish_plot(output_dir / "matrix_size_sweep.png")


def plot_thread_count_sweep(df: pd.DataFrame, output_dir: Path) -> None:
    data = df[df["sweep"] == "thread_count"].copy()

    plt.figure(figsize=(9, 5.2))
    for kernel in KERNEL_ORDER:
        subset = data[data["kernel"] == kernel].sort_values("threads")
        if subset.empty:
            continue
        plt.plot(subset["threads"], subset["time_sec"], marker="o", label=kernel)

    plt.yscale("log")
    plt.xlabel("Threads")
    plt.ylabel("Median time (seconds, log scale)")
    plt.title("Thread count sweep, N=512")
    _finish_plot(output_dir / "thread_count_sweep.png")


def plot_block_size_sweep(df: pd.DataFrame, output_dir: Path) -> None:
    data = df[df["sweep"] == "block_size"].copy()

    plt.figure(figsize=(9, 5.2))
    for kernel in KERNEL_ORDER:
        subset = data[data["kernel"] == kernel].sort_values("block_size")
        if subset.empty:
            continue
        plt.plot(subset["block_size"], subset["time_sec"], marker="o", label=kernel)

    plt.yscale("log")
    plt.xlabel("Block size")
    plt.ylabel("Median time (seconds, log scale)")
    plt.title("Block size sweep, N=512, threads=1")
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
