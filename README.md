# Parallel Matrix Multiplication

A learning-oriented ML kernel project that implements and benchmarks matrix multiplication and related tensor operations from scratch.

## Features

- Dynamic matrix allocation
- Naive matrix multiplication
- Row-partitioned pthread matrix multiplication
- Median-based benchmarking

## Build

```bash
make
```

## Run

```bash
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

## Example

```text
[benchmark]
A: 1000x1000, B: 1000x1000, threads: 10, iterations: 10
naive median:    ...
threaded median: ...
speedup:         ...
```
