# Repository Guidelines

## Project Overview

`tinykernels` is a C99 educational project exploring CPU matrix-multiplication kernels and, increasingly, a mini Qwen3-0.6B LLM inference engine. It implements multiple matmul algorithm variants — naive scalar, loop-order (IJK/IKJ), cache blocking, SSE SIMD (IKJ only), pthread parallelism, and optional OpenMP — plus a GGUF reader, byte-level BPE tokenizer, and model-config loader, along with correctness tests, CSV benchmarking, and plotting.

The codebase uses a single unified `tk_` namespace grouped by domain, a `tk_status` error enum, and `tk_x*` abort-on-OOM allocators (see the Status Contract below).

## Architecture & Data Flow

```
src/main.c  (thin CLI, subcommand dispatch)
  ├── test   → src/test/tk_test.c        (matmul + tokenizer + gguf/model regressions)
  ├── bench  → src/bench/tk_bench.c      (timing sweeps, 14-col CSV)
  ├── model  → src/model/tk_model.c + tk_gguf.c (load/validate/spot-check)
  ├── tokenize/detokenize → src/model/tk_tokenizer.c
  ├── forward/generate    → src/model/tk_forward.c + tk_generate.c (KV cache, sampling)
  └── bench-infer         → real inference timing via tk_infer_set_cfg

Dependency direction is inward, no upward deps:
  app(main) → test/bench → backend/kernels → model → core(tk_common, tk_matrix, tk_util)

Core API (src/backend/tk_matmul.c)
  ├── tk_mat_mul()      → allocates output, delegates to tk_mat_mul_into()
  └── tk_mat_mul_into() → validates, zeroes output, dispatches to backend

Backend dispatch (switch on tk_matmul_cfg.backend)
  ├── single  → src/backend/tk_single.c  (serial, calls tk_kernel_run)
  ├── pthread → src/backend/tk_pthread.c (row-split parallel, TkErr_INTERNAL on failure)
  └── openmp  → src/backend/tk_openmp.c  (#pragma omp parallel for)

Kernel dispatch (include/internal/tk_kernel_ops.h)
  └── tk_kernel_run()        → selects SIMD or scalar based on config+CPU features
      ├── SIMD (src/kernels/tk_simd.c)  → SSE __m128, 4-wide unroll, IKJ only
      └── scalar (include/internal/tk_kernel_ops.h)
          ├── blocking enabled  → tk_kernel_ijk_blocked or tk_kernel_ikj_blocked
          └── no blocking       → tk_kernel_ijk or tk_kernel_ikj
      └── Loop kernels are `static inline` header-only

Validation (src/backend/tk_validate.c)
  └── tk_mat_validate / tk_mat_input_valid / tk_matmul_cfg_valid → tk_status

Model (src/model/)
  ├── tk_gguf.c       GGUF v3 reader (header, metadata KV, tensor info, F32/Q8_0 dequant)
  ├── tk_tokenizer.c  byte-level BPE tokenizer/detokenizer (vocab from GGUF)
  ├── tk_model.c      TkModel config (hardcoded Qwen3-0.6B + GGUF-derived)
  ├── tk_forward.c    TkInfer: per-layer KV cache + cached resident weights
  └── tk_generate.c   chat template + sampling (temperature/top-k/top-p)

TkInfer dequantizes **every weight resident** at init (transposed to y = x @ W),
so the per-token step does zero file I/O; the hot GEMMs (QKV, output, SwiGLU
MLP, logits) run the SIMD IKJ kernel by default (config swapped via
tk_infer_set_cfg). bench-infer times tokens across plain/blocked/fast configs.

Data layout: Row-major `float` matrices. `tk_elem_t` is `typedef float`.
```

## Key Directories

|Path|Purpose|
|---|---|
|`include/`|Public headers: `tk_common.h`, `tk_matrix.h`, `tk_backend.h`, `tk_kernels.h`, `tk_gguf.h`, `tk_tokenizer.h`, `tk_model.h`, `tk_bench.h`, `tk_test.h`|
|`include/internal/`|Private headers (never used outside the build): `tk_kernel_ops.h`, `tk_backend_impl.h`, `unicode_ranges.h`|
|`src/core/`|Foundation: `tk_util.c` (alloc/logger/`tk_now_seconds`), `tk_matrix.c`|
|`src/kernels/`|`tk_simd.c` — SSE IKJ kernels (`__m128` vectorized)|
|`src/backend/`|Matmul engine: `tk_matmul.c`, `tk_single.c`, `tk_pthread.c`, `tk_openmp.c`, `tk_validate.c`|
|`src/model/`|GGUF-derived: `tk_gguf.c`, `tk_tokenizer.c`, `tk_model.c`|
|`src/test/`|`tk_test.c` — correctness + regression harness|
|`src/bench/`|`tk_bench.c` — benchmark sweeps, CSV writer|
|`scripts/`|`plot_benchmarks.py` — matplotlib plots from benchmark CSV. Unchanged|
|`results/data/`|`benchmark_results.csv` — auto-generated benchmark data|
|`results/plots/`|Auto-generated PNG charts|

## Development Commands

```bash
make                      # build (O3, -march=native, strict -Werror set)
make debug                # build with -O0 -g3
make sanitize             # build with -fsanitize=address,undefined
./tinykernels test        # run correctness suite
./tinykernels bench       # run benchmark suite (requires results/ dirs)
./tinykernels all         # tests + benchmarks
make bench                # run benchmarks + regenerate plots
make plots                # regenerate plots from existing CSV
make format               # clang-format all C/H sources (replaces old format.sh)
make clean                # remove build artifacts
```

Enable OpenMP: `OPENMP=1 make clean && make`

Plots need `matplotlib`/`pandas`; `make bench` prefers the project venv (`.venv/bin/python3`) when present, else falls back to `python3`.

## Status Contract

- **Allocation**: `tk_xmalloc`/`tk_xcalloc`/`tk_xrealloc`/`tk_xstrdup` abort (`exit`) on OOM; callers never branch on allocation NULL. Only `free()` is used for release.
- **Errors**: one `tk_status` enum (`TK_OK`, `TK_ERR_ARG`, `TK_ERR_IO`, `TK_ERR_FORMAT`, `TK_ERR_UNSUPPORTED`, `TK_ERR_INTERNAL`). Fail-capable operations return `tk_status` and `TK_LOGE` exactly once at the point of failure. Pointer-returning constructors (`tk_gguf_open`, `tk_tokenizer_open`) return NULL + `TK_LOGE`; `tk_mat_new` returns an empty sentinel + `TK_LOGE`. Infallible ops are `void`/getters that return defaults on absence.
- **Diagnostics** go to stderr (`tk_log_error`/`TK_LOGE`/`tk_log_info`), never stdout — stdout is reserved for CLI contract output (tokenize ids, detokenize text, bench CSV).

## Code Conventions & Common Patterns

- **C99 strict**: `-std=c99 -Wall -Wextra -Wpedantic -Werror` plus `-Wshadow -Wformat=2 -Wstrict-prototypes -Wmissing-prototypes -Wwrite-strings -Wundef -Wpointer-arith`. No C++ features, no GNU extensions.
- **Boolean flags**: Use `_Bool`/`bool` (via `<stdbool.h>`), never `int` for true/false. `tk_matmul_cfg` fields `use_blocking`, `use_simd` are `bool`.
- **Const-correct**: Input matrices are `const TkMatrix *`, output is `TkMatrix *`.
- **Static inline**: Loop kernels and `tk_kernel_run` dispatch are `static inline` in headers for zero-call overhead; SSE kernels are declared there, defined in `src/kernels/tk_simd.c`.
- **`#if ENABLE_OPENMP` / `#if defined(__SSE__)`**: Feature-gated code paths with `(void)` casts for unused params in disabled branches; disabled branches return `TK_ERR_UNSUPPORTED`.
- **`-Wmissing-prototypes`**: every non-static function is declared in a header the defining `.c` includes.
- **Naming**: `tk_` prefix for everything, grouped by domain (`tk_mat_*` matrices, `tk_kernel_*` kernels, `tk_backend_*` backends, `tk_gguf_*` GGUF, `tk_tokenize*`/`tk_detokenize*`, `tk_model_*`). Zero-arg functions are declared `(void)`.
- **Size types**: `size_t` throughout. `SIZE_MAX` from `<stdint.h>` for overflow checks.

## Important Files

|File|Role|
|---|---|
|`src/main.c`|Entry point, subcommand dispatch (`test`/`bench`/`all`/`model`/`tokenize`/`detokenize`/`forward`/`generate`/`bench-infer`, default `test`)|
|`include/tk_common.h`|Foundation: `tk_status`, `tk_x*` alloc, logger, `tk_min`, `tk_now_seconds`|
|`src/core/tk_util.c`|Definitions for the foundation (alloc, logger, `tk_now_seconds`)|
|`include/tk_matrix.h`, `src/core/tk_matrix.c`|`TkMatrix` struct + `tk_mat_*` API|
|`include/tk_backend.h`|Public matmul API: `tk_matmul_cfg`, `tk_backend`, `tk_loop`, `tk_mat_mul*`, validation decls|
|`src/backend/tk_matmul.c`|Dispatch + config plumbing (`tk_mat_mul`, `tk_mat_mul_into`, `tk_matmul_cfg_new`, labels)|
|`include/internal/tk_kernel_ops.h`|`tk_kernel_run` + `static inline` loop kernels + SSE decls|
|`src/kernels/tk_simd.c`|SSE IKJ kernels (`__m128` vectorized)|
|`include/tk_gguf.h`, `src/model/tk_gguf.c`|GGUF v3 reader: header + metadata KV + tensor info, on-demand F32/Q8_0 dequant; shared by tokenizer and weight loading|
|`include/tk_tokenizer.h`, `src/model/tk_tokenizer.c`|Byte-level BPE tokenizer/detokenizer (vocab from GGUF)|
|`include/tk_model.h`, `src/model/tk_model.c`|`TkModel` config (hardcoded + GGUF-derived) + config helpers|
|`include/tk_forward.h`, `src/model/tk_forward.c`|`TkInfer` incremental engine: per-layer KV cache, resident dequantized weights, `tk_infer_step` + `tk_infer_set_cfg` GEMM override|
|`include/tk_generate.h`, `src/model/tk_generate.c`|Chat template + sampling (temperature/top-k/top-p, seeded xorshift64)|
|`src/test/tk_test.c`|Correctness vs reference (1e-6) + tokenizer + gguf/model regression cases|
|`src/bench/tk_bench.c`|Benchmark sweeps, CSV writer (`tk_bench_default_suite`)|
|`scripts/plot_benchmarks.py`|matplotlib plots: matrix size, thread count, block size sweeps. Unchanged|
|`Makefile`|Build system, strict flags, sanitizers, OpenMP toggle, plots (`make format`)|
|`compile_flags.txt`|LSP/clangd config (`-Iinclude`)|

## Runtime/Tooling Preferences

- **Compiler**: GCC (Clang should work). Requires `-std=c99`.
- **OpenMP**: Disabled by default. Enable with `OPENMP=1` flag. Requires compiler with OpenMP support (`-fopenmp`).
- **SIMD**: Auto-detected at compile time via `__SSE__` macro. SIMD kernels silently disabled if target lacks SSE. `tk_simd.c` compiles clean under `-Wpedantic -Werror`, so no per-file `-Wno-pedantic` needed.
- **Python**: `scripts/plot_benchmarks.py` requires `matplotlib` and `pandas`; `scripts/verify_tokenizer.py` runs with plain stdlib. Both prefer `.venv/bin/python3` when present.
- **No external C dependencies**: Only `<stdio.h>`, `<stdlib.h>`, `<pthread.h>`, `<omp.h>` (optional).

## Testing & QA

- **Correctness**: `./tinykernels test` — `tk_test_all()` runs the matmul suite (each config vs a single-backend/IJK/no-blocking/no-SIMD reference with `1e-6` float tolerance, dimensions 1×1 to 70×70, threads 1–8, block sizes 1–128), plus tokenizer and gguf/model regressions. Returns exit 1 on any failure.
- **Regression anchors (must not "fix" by changing the expectation)**: matmul 1e-6 tolerance; tokenizer id list `[9707,11,1879,0,1096,374,264,1207,16948,18,1273,13]` for "Hello, world! This is a Qwen3 test."; `token_embd.weight[0] = -0.00930309296f`; vocab 151936, layers 28.
- **Sanitizers**: `make sanitize && ./tinykernels test` and `./tinykernels model` — catches memory errors (ASan) and UB (UBSan). Afterward a plain `make` needs `make clean && make` to drop the sanitizer objects (unchanged behavior).
- **Benchmark output**: CSV at `results/data/benchmark_results.csv` with 14 columns (`sweep,rows,inner,cols,backend,loop_order,use_blocking,use_simd,num_threads,block_size,iterations,time_sec,speedup_vs_baseline,label`). Plots at `results/plots/`.
- **No test framework**: Custom inline harness in `src/test/tk_test.c` with ANSI-colored output.
