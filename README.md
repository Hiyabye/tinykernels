# tinykernels

`tinykernels` is a learning-oriented CPU kernel project for implementing and benchmarking matrix multiplication optimizations from scratch.

The project starts from a naive matrix multiplication implementation and progressively adds loop-order optimization, blocking, pthread parallelism, and OpenMP parallelism. The goal is not to beat vendor libraries; the goal is to understand how low-level implementation choices affect performance.

## Current scope

- Dense row-major matrix allocation
- Config-driven matrix multiplication API
- Output-buffer API via `matmul_into()`
- Single-threaded kernels
- pthread row-partitioned kernels
- OpenMP row-parallel kernels
- IJK and IKJ loop orders
- Optional blocking/tiling
- Correctness tests across matrix shapes, thread counts, and block sizes
- Median-based benchmark CSV export
- Matplotlib benchmark plotting script

## Project layout

```text
include/
  bench.h
  matmul.h
  matrix.h
  test.h
src/
  main.c
  matrix.c
  bench.c
  test.c
  matmul/
    matmul.c        # public dispatch and validation
    kernels.h       # shared internal kernel helpers
    single.c        # single-thread backend
    pthread.c       # pthread backend
    openmp.c        # OpenMP backend
scripts/
  plot_benchmarks.py
assets/
  matrix_size_sweep.png
  thread_count_sweep.png
  block_size_sweep.png
```

## Matmul configuration

Instead of one enum per kernel, `tinykernels` uses composable configuration flags:

```c
typedef struct {
  MatmulBackend backend;      // single, pthread, openmp
  MatmulLoopOrder loop_order; // ijk, ikj
  int use_blocking;           // 0 or 1
  size_t num_threads;
  size_t block_size;
} MatmulConfig;
```

This makes combinations explicit. For example:

| Backend | Loop order | Blocking | Label |
|---|---|---:|---|
| single | IJK | off | `single_plain_ijk` |
| single | IKJ | on | `single_blocked_ikj` |
| pthread | IKJ | off | `pthread_plain_ikj` |
| pthread | IKJ | on | `pthread_blocked_ikj` |
| openmp | IKJ | on | `openmp_blocked_ikj` |

## API

```c
Matrix matmul(const Matrix *a, const Matrix *b, MatmulConfig cfg);
int matmul_into(const Matrix *a, const Matrix *b, Matrix *c, MatmulConfig cfg);
```

`matmul()` is a convenience wrapper that allocates the output matrix. `matmul_into()` writes into an existing output matrix and is better suited for benchmarking and future CUDA-style APIs.

## Benchmark results

The default benchmark suite writes `benchmark_results.csv` and uses median runtime over repeated runs.

### Matrix size sweep

<img src="assets/matrix_size_sweep.png" width="70%">

### Thread count sweep

<img src="assets/thread_count_sweep.png" width="70%">

### Block size sweep

<img src="assets/block_size_sweep.png" width="70%">

## Build and run

```bash
make run
```

To regenerate plots after benchmarking:

```bash
make plots
```

## Sanitizer build

```bash
make sanitize
./tinykernels
```

## OpenMP notes

OpenMP is enabled by default:

```bash
make
```

To build without OpenMP:

```bash
make OPENMP=0
```

On macOS with Homebrew LLVM, you can pass an LLVM prefix:

```bash
make LLVM_PREFIX=/opt/homebrew/Cellar/llvm/22.1.6
```

Expected CSV schema:

```text
sweep,n,threads,block_size,iterations,kernel,time_sec,speedup_vs_ref
```

## Roadmap

- Add quick/full benchmark modes via CLI arguments
- Add benchmark environment metadata to README
- Add profiling notes using `perf`, Instruments, or similar tools
- Explore `float` vs `double`
- Explore SIMD/vectorization
- Port core abstractions to C++ with RAII
- Add CUDA kernels
- Extend toward small ML primitives such as linear layers, activation kernels, and normalization kernels
