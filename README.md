# Parallel Matrix Multiplication

A learning-oriented CPU kernel project for implementing and benchmarking matrix multiplication optimizations from scratch.

## Current scope

- Dense matrix allocation
- Reference IJK matrix multiplication
- Cache-friendly IKJ multiplication
- Blocked matrix multiplication
- Row-partitioned pthread multiplication
- Median-based benchmarking

## Roadmap

- C++ port with RAII
- SIMD/vectorization
- CUDA backend
- Linear layer forward pass
- Activation and normalization kernels

## Build & Run

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
