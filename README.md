# tinykernels

`tinykernels` is a small C project for learning how matrix multiplication kernels work on the CPU.

It includes a naive baseline, loop-order variants, blocking, SIMD IKJ kernels, pthread parallelism, optional OpenMP, correctness tests, CSV benchmarks, and a plotting script.

## Build

```bash
make
```

Run correctness tests:

```bash
make test
```

Run benchmarks and regenerate plots:

```bash
make bench
```

Build with sanitizers:

```bash
make sanitize
./tinykernels test
```

OpenMP is disabled by default. Enable it when your compiler supports OpenMP:

```bash
make OPENMP=1
```


SIMD kernels are enabled automatically when the target supports x86 SSE. Unsupported SIMD requests are rejected instead of silently falling back to scalar code.

## CLI

```bash
./tinykernels test   # correctness tests
./tinykernels bench  # benchmark suite
./tinykernels all    # tests, then benchmarks
./tinykernels model  # print GGUF-derived architecture, validate tensor inventory
./tinykernels tokenize "text"    # byte-level BPE -> token ids
./tinykernels detokenize 9707 11 # token ids -> text
./tinykernels forward "prompt"   # full Qwen3 forward pass -> greedy next-token ids + logits
./tinykernels generate "prompt"  # autoregressive generation (sampling + KV cache)
```

Running `./tinykernels` with no arguments defaults to `test`.

`forward` runs the whole Qwen3-0.6B transformer (embedding, RMSNorm, QKV +
QK-norm, RoPE, GQA attention, SwiGLU MLP, tied lm_head) on top of the matmul
kernels, writes full per-token logits to `results/data/forward_logits.bin`, and
prints the greedy next-token id per position. Its logits are cross-checked
against an independent numpy reference and llama.cpp via
`scripts/verify_forward.py`.

`generate` runs the same pass incrementally with a per-layer KV cache and an
autoregressive loop, applying temperature/top-k/top-p sampling plus the Qwen3
chat template (with `--think` to start a reasoning block). Sampling is
deterministic for a fixed `--seed`.

## API

```c
MatmulConfig cfg = matmul_config(
    MATMUL_BACKEND_PTHREAD,
    MATMUL_LOOP_IKJ,
    1,   // use blocking
    0,   // use SIMD
    4,   // threads
    32   // block size
);

Matrix c = matmul(&a, &b, cfg);
```

Use `matmul_into()` when you already own the output buffer, especially for benchmarks:

```c
int ok = matmul_into(&a, &b, &c, cfg);
```

## Project Layout

```text
include/              public headers
src/matmul/           matmul dispatch and backends
results/              benchmark data, CSV, and plots
scripts/              benchmark plotting
```

## Benchmarks

The following data is from an AMD Ryzen 7 9800X3D with NVIDIA GeForce RTX 5070 Ti, running Arch Linux.

<img src="results/plots/matrix_size_sweep.png" width="70%">

<img src="results/plots/thread_count_sweep.png" width="70%">

<img src="results/plots/block_size_sweep.png" width="70%">

CSV columns:

```text
sweep,rows,inner,cols,backend,loop_order,use_blocking,use_simd,num_threads,block_size,iterations,time_sec,speedup_vs_baseline,label
```
