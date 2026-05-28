# tinykernels

A learning-oriented CPU kernel project for implementing and benchmarking matrix multiplication optimizations from scratch.

## Overview

`tinykernels` starts from a simple reference matrix multiplication kernel and incrementally applies CPU-side optimization techniques such as cache-friendly loop ordering, blocking, pthread-based row partitioning, and OpenMP parallelism.

The goal is not to beat vendor libraries. The goal is to understand how low-level implementation choices affect performance.

## Implemented kernels

| Kernel | Description |
|:---:|:---:|
| `MATMUL_REF_IJK` | Reference `i-j-k` implementation |
| `MATMUL_SEQ_IKJ` | Cache-friendlier sequential `i-k-j` loop order |
| `MATMUL_SEQ_BLOCKED_IKJ` | Sequential blocked/tiled `i-k-j` implementation |
| `MATMUL_PAR_ROWS_IJK` | pthread row-partitioned `i-j-k` implementation |
| `MATMUL_PAR_ROWS_BLOCKED_IKJ` | pthread row-partitioned blocked `i-k-j` implementation |
| `MATMUL_OPENMP_IKJ` | OpenMP parallel `i-k-j` implementation |

## Benchmark results

The benchmark uses median runtime over repeated runs. The current benchmark suite checks matrix-size scaling, thread-count scaling, and block-size sensitivity.

### Matrix size sweep

![Matrix size sweep](assets/matrix_size_sweep.png){: width="50%" height="50%"}

### Thread count sweep

![Thread count sweep](assets/thread_count_sweep.png){: width="50%" height="50%"}

### Block size sweep

![Block size sweep](assets/block_size_sweep.png){: width="50%" height="50%"}

## Observations

- `Sequential IKJ` is much faster than the reference `IJK` implementation because it improves row-major memory locality.
- pthread row partitioning avoids write races by assigning disjoint row ranges of `C` to each worker.
- `Parallel Rows Blocked IKJ` and `OpenMP IKJ` show strong speedups on the 512×512 benchmark.
- Block size matters: smaller blocks are not automatically faster, and the best value depends on the machine and cache behavior.

## Build & run

```bash
make
make run
```

## Debug build

```bash
make debug
```

## Sanitizer build

```bash
make sanitize
make run
```

## Plot benchmark graphs

Save benchmark results as CSV using this schema:

```text
sweep,n,threads,block_size,iterations,kernel,time_sec,speedup_vs_ref
```

Then run:

```bash
python3 scripts/plot_benchmarks.py benchmark_results.csv assets
```

## Roadmap

- Add `matmul_into()` to separate output allocation from compute timing
- Add CSV benchmark export
- Add profiling notes using `perf`, Instruments, or similar tools
- Explore SIMD/vectorization
- Port core abstractions to C++
- Add CUDA kernels
- Extend toward small ML primitives such as linear layers, activation kernels, and normalization kernels
