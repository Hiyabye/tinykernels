# Repository Guidelines

## Project Overview

**tinykernels** is a C99 project for learning and comparing CPU matrix multiplication kernel implementations. It provides a unified dispatch API over multiple backends (single-threaded, pthread, OpenMP), loop-order variants (IJK, IKJ), blocking optimization, and SIMD acceleration (x86 SSE). Outputs are correctness tests, CSV benchmark data, and matplotlib plots.

## Architecture & Data Flow

```
main.c (CLI entry)
  ├── "test"  → test_matmul.h → verifies kernel correctness
  ├── "bench" → bench_matmul.h → runs sweeps, writes CSV
  └── "all"   → tests + bench

include/matrix.h      → Matrix type, element allocator
include/matmul.h      → public API, MatmulConfig, dispatch
include/kernels.h       → kernel dispatch (inline + SIMD selectors)
src/matmul/matmul.c   → validation + backend dispatch switch
src/matmul/single.c   → single-threaded kernel launcher
src/matmul/pthread.c  → pthread row-split parallelism
src/matmul/openmp.c   → OpenMP parallel for with blocking
src/matrix.c          → matrix alloc/fill/print free
src/bench_matmul.c     → benchmark harness, CSV output, plots
scripts/              → Python plotting script
```

**Kernel dispatch chain:**
`matmul_into()` → `validate_*()` → switch(backend) → `tk_matmul_*_into()` → `tk_matmul_range()` → selects SIMD vs scalar → selects loop order → selects blocking

**SIMD selection:** at compile-time via `TK_HAVE_SSE` macro; runtime rejected via `matmul_simd_available()` guard if target lacks support.

## Key Directories

| Path | Purpose |
|------|---------|
| `src/` | Core source — `main.c`, `matrix.c`, `bench_matmul.c`, `matmul/` |
| `src/matmul/` | Backend implementations: `single.c`, `pthread.c`, `openmp.c`, `simd.c`, `matmul.c` |
| `include/` | Public headers + test harness: `matrix.h`, `matmul.h`, `kernels.h`, `bench_matmul.h`, `test_matmul.h` |
| `results/` | Generated benchmark data CSV and plot PNGs |
| `scripts/` | `plot_benchmarks.py` — matplotlib plots from CSV |

## Development Commands

```bash
make              # Build (release: -O3 -march=native)
make test         # Run correctness tests
make bench        # Run benchmarks + regenerate plots
make all          # Tests then benchmarks
make run          # Execute binary with no args (defaults to "test")
make debug        # Rebuild with -O0 -g3
make sanitize     # Build with ASAN + UBSAN, then run tests
make clean        # Remove build/ and binary

make OPENMP=1                      # Enable OpenMP backend
```

## Code Conventions & Common Patterns

**Formatting:** `.clang-format` uses LLVM style, 2-space indent, no tabs, column limit 120. Short loops, ifs/elsifs, and blocks may stay on one line.

**Naming:**
- Kernel functions: `tk_matmul_<backend|variant>_<into>` (e.g. `tk_matmul_range_ikj`, `tk_matmul_pthread_into`)
- Matrix API: `matrix_new`, `matrix_free`, `matrix_fill`, `matrix_fill_pattern`, `matrix_print`
- Config: `matmul_config()` constructor; `matmul_config_label()` for string labels
- Private helpers: `static` in `matmul.c` — `validate_matrices`, `validate_input_matrices`, `validate_config`

**Error handling:** Return `0` / `NULL` on failure; `1` / `Matrix` on success. Errors logged via `fprintf(stderr, ...)`. No exceptions or goto cleanup.

**Memory:** `Matrix` owns a `calloc`'d `data` pointer. `matrix_new()` checks for size overflow before allocation. `matrix_free()` nulls out the pointer after free.

**Parallelism:**
- **pthread:** row-split with uneven load balancing (extra rows distributed to first workers). `struct PthreadArgs` passed per-thread. Manual join loop with partial cleanup on error.
- **OpenMP:** `#pragma omp parallel for` with static scheduling. Blocking variant uses `schedule(static)` on outer block loop; non-blocking on per-row loop.
 - **SIMD:** SSE (`xmmintrin.h`) kernels only on IKJ loop order. Uses `_Static_assert` guard on `mat_elem_t` being `float`. Dead-code elimination on unsupported targets.

**Include guards:** Standard `#ifndef`/`#define`/`#endif` pattern with descriptive names (`TINYKERNELS_MATMUL_KERNELS_H`).

**Type aliases:** `mat_elem_t` is `float`. All sizes use `size_t`; no `int` for dimensions.

## Important Files

| File | Role |
|------|------|
| `src/main.c` | CLI entry point — parses `test|bench|all` |
| `include/matmul.h` | Public API — `MatmulConfig`, `matmul()`, `matmul_into()` |
| `include/matrix.h` | `Matrix` type definition |
| `src/matmul/matmul.c` | Validation, config construction, backend dispatch |
| `include/kernels.h` | All kernel implementations (inline scalar + SIMD extern) |
| `src/matmul/pthread.c` | pthread parallel backend |
| `src/matmul/openmp.c` | OpenMP parallel backend |
| `Makefile` | Build system — targets, flags, OpenMP toggle, sanitizer builds |
| `include/test_matmul.h` | Correctness test suite |
| `src/bench_matmul.c` | Benchmark harness — 3 sweeps: matrix size, thread count, block size |

## Runtime / Tooling Preferences

- **Compiler:** GCC or Clang, C99 (`-std=gnu99`). POSIX threads required.
- **OpenMP:** Optional, disabled by default. Enable with `make OPENMP=1`.
- **SIMD:** Auto-detected at compile time (`__SSE__` → SSE). No runtime fallback — unsupported SIMD requests are rejected.
- **Python:** `plot_benchmarks.py` requires `matplotlib` and `pandas` (run via `python3`).
- **No package manager:** Pure Makefile build. No dependencies beyond standard C library, pthreads, and optional libomp.

## Testing & QA

- **Correctness tests:** `make test` runs `test_matmul.h` which validates kernel output against expected results.
- **Sanitizers:** `make sanitize` rebuilds with `-fsanitize=address,undefined` and runs tests. Use to catch memory bugs.
- **Benchmarks:** `make bench` runs three sweeps (matrix size, thread count, block size) writing CSV to `results/data/benchmark_results.csv` and plots to `results/plots/`.
- **CSV schema:** `sweep,rows,inner,cols,backend,loop_order,use_blocking,use_simd,num_threads,block_size,iterations,time_sec,speedup_vs_baseline,label`

## Architecture Notes for AI Assistants

- **Do NOT add new backends without understanding the dispatch chain.** The switch in `matmul.c` line 103 routes to three backends defined in `kernels.h`.
- **SIMD kernels require IKJ loop order** (enforced at line 90-92 of `matmul.c`). Adding SIMD support for IJK requires kernel rewrite.
- **Blocking only works on row-parallel backends.** OpenMP blocks on the outer loop; pthreads block on row-split ranges.
- **All kernel functions are row-range scoped** — `row_start`/`row_end` parameters. This is the unit of parallelism.
- **`matmul_into()` zeroes the output buffer** before dispatch; `matmul()` allocates and calls `into()`.
- **Config validation is eager** — `matmul_into()` and `matmul()` return failure before any work if config is invalid.
