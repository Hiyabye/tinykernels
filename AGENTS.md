# Repository Guidelines

## Project Overview

`tinykernels` is a C99 educational project exploring CPU matrix-multiplication kernels. It implements multiple algorithm variants — naive scalar, loop-order (IJK/IKJ), cache blocking, SSE SIMD (IKJ only), pthread parallelism, and optional OpenMP — along with correctness tests, CSV benchmarking, and plotting.

## Architecture & Data Flow

```
main.c
  ├── test mode     → test_matmul.c    (reference-matrix comparison)
  ├── bench mode    → bench_matmul.c   (timing sweeps, CSV output)
  └── all mode      → both

Core API (src/matmul/matmul.c)
  ├── matmul()      → allocates output, delegates to matmul_into()
  └── matmul_into() → validates, zeroes output, dispatches to backend

Backend dispatch (switch on MatmulConfig.backend)
  ├── single        → src/matmul/single.c    (serial, calls matmul_range)
  ├── pthread       → src/matmul/pthread.c   (row-split parallel)
  └── openmp        → src/matmul/openmp.c    (#pragma omp parallel for)

Kernel dispatch (include/kernels/dispatch.h)
  └── matmul_range()          → selects SIMD or scalar based on config+CPU features
      ├── SIMD (src/matmul/simd.c)  → SSE __m128, 4-wide unroll, IKJ only
      └── scalar (include/kernels/dispatch.h)
          ├── blocking enabled  → blocked_ijk or blocked_ikj
          └── no blocking       → ijk or ikj
      └── Loop kernels (include/kernels/loop.h) — static inline header-only

Validation (src/matmul/validate.c)
  └── validate_matrices / validate_input_matrices / validate_config

Data layout: Row-major `float` matrices. `mat_elem_t` is `typedef float`.
```

## Key Directories

|Path|Purpose|
|---|---|
|`include/`|Public headers: `matmul.h`, `matrix.h`, `bench_matmul.h`, `test_matmul.h`|
|`include/kernels/`|Internal kernel headers: `loop.h`, `simd.h`, `dispatch.h`, `backends.h`|
|`src/matmul/`|Matmul implementation: `matmul.c`, `single.c`, `simd.c`, `pthread.c`, `openmp.c`, `validate.c`|
|`src/`|Root source: `main.c`, `matrix.c`, `bench_matmul.c`, `test_matmul.c`|
|`scripts/`|`plot_benchmarks.py` — matplotlib plots from benchmark CSV|
|`results/data/`|`benchmark_results.csv` — auto-generated benchmark data|
|`results/plots/`|Auto-generated PNG charts|

## Development Commands

```bash
make                      # build (O3, -march=native)
make debug                # build with -O0 -g3
make sanitize             # build with -fsanitize=address,undefined
./tinykernels test        # run correctness suite
./tinykernels bench       # run benchmark suite (requires results/ dirs)
./tinykernels all         # tests + benchmarks
make bench                # run benchmarks + regenerate plots
make plots                # regenerate plots from existing CSV
make clean                # remove build artifacts
```

Enable OpenMP: `OPENMP=1 make clean && make`

## Code Conventions & Common Patterns

- **C99 strict**: `-std=c99 -Wall -Wextra -Wpedantic`. No C++ features, no GNU extensions.
- **Boolean flags**: Use `_Bool`/`bool` (via `<stdbool.h>`), never `int` for true/false. `MatmulConfig` fields `use_blocking`, `use_simd` are `bool`.
- **Error reporting**: `perror()` for system call failures, `fprintf(stderr, ...)` for application-level errors. Debug/diagnostic output goes to stderr, never stdout.
- **Const-correct**: Input matrices are `const Matrix *`, output is `Matrix *`.
- **Static inline**: Loop kernels and dispatch logic are `static inline` in headers for zero-call overhead.
- **`#if ENABLE_OPENMP` / `#if defined(__SSE__)`**: Feature-gated code paths with `(void)` casts for unused params in disabled branches.
- **Naming**: `matmul_*` prefix for all public matmul API. `matrix_*` for matrix ops. `validate_*` for validation. `bench_*` for benchmarking.
- **No raw `extern`**: Validation function declarations live in `include/kernels/backends.h`, transitively included via `kernels.h`.
- **`mat_elem_t`**: Aliased to `float`. SIMD kernels use `__m128` (4×float).
- **Size types**: `size_t` throughout. `SIZE_MAX` from `<stdint.h>` for overflow checks.

## Important Files

|File|Role|
|---|---|
|`src/main.c`|Entry point, CLI routing (`test`/`bench`/`all`, default `test`)|
|`src/matmul/matmul.c`|Core API: `matmul()`, `matmul_into()`, `matmul_config()`, `matmul_simd_available()`|
|`include/matmul.h`|Public API: `MatmulConfig`, `MatmulBackend`, `MatmulLoopOrder`, function declarations|
|`include/matrix.h`|`Matrix` struct, `matrix_*` API|
|`include/kernels/dispatch.h`|`matmul_range()` — selects SIMD vs scalar, blocking vs plain|
|`include/kernels/loop.h`|`static inline` loop kernels: ijk, ikj, blocked variants|
|`include/kernels/backends.h`|Backend declarations + validation function declarations|
|`src/matmul/simd.c`|SSE IKJ kernels (`__m128` vectorized)|
|`include/gguf.h`, `src/llm/gguf.c`|GGUF v3 reader: header + metadata KV + tensor info, on-demand F32/Q8_0 dequant; shared by the tokenizer and weight loading|
|`include/qwen.h`, `src/llm/qwen.c`|`QwenConfig` (hardcoded + GGUF-derived) and config helpers|
|`src/llm/qwen_tokenizer.c`|Byte-level BPE tokenizer/detokenizer (vocab from GGUF via `gguf.h`)|
|`src/test_matmul.c`|Correctness tests vs reference output (1e-6 tolerance)|
|`scripts/plot_benchmarks.py`|matplotlib plots: matrix size sweep, thread count sweep, block size sweep|
|`Makefile`|Build system, flags, sanitizers, OpenMP toggle|
|`compile_flags.txt`|LSP/clangd config (`-Iinclude`)|

## Runtime/Tooling Preferences

- **Compiler**: GCC (Clang should work). Requires `-std=c99`.
- **OpenMP**: Disabled by default. Enable with `OPENMP=1` flag. Requires compiler with OpenMP support (`-fopenmp`).
- **SIMD**: Auto-detected at compile time via `__SSE__` macro. SIMD kernels silently disabled if target lacks SSE.
- **Python**: `scripts/plot_benchmarks.py` requires `matplotlib` and `pandas`.
- **No external C dependencies**: Only `<stdio.h>`, `<stdlib.h>`, `<pthread.h>`, `<omp.h>` (optional).

## Testing & QA

- **Correctness**: `./tinykernels test` — compares each config output against a reference `matmul()` (single backend, IJK, no blocking, no SIMD) with `1e-6` float tolerance. Tests various matrix dimensions (1×1 to 70×70) and config sweeps (threads 1–8, block sizes 1–128). Returns exit 1 on failure.
- **Sanitizers**: `make sanitize && ./tinykernels test` — catches memory errors (ASan) and UB (UBSan).
- **Benchmark output**: CSV at `results/data/benchmark_results.csv` with 14 columns (`sweep,rows,inner,cols,backend,loop_order,use_blocking,use_simd,num_threads,block_size,iterations,time_sec,speedup_vs_baseline,label`). Plots at `results/plots/`.
- **No test framework**: Custom inline test harness in `src/test_matmul.c` with ANSI-colored output.
