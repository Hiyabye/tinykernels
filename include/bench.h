#ifndef BENCH_H
#define BENCH_H

#include <stddef.h>

// Benchmarking function to test matrix multiplication performance
double naive_benchmark(size_t rows, size_t inner, size_t cols,
                       size_t iterations);
double threaded_benchmark(size_t rows, size_t inner, size_t cols,
                          size_t num_threads, size_t iterations);
double ikj_benchmark(size_t rows, size_t inner, size_t cols, size_t iterations);
double blocked_benchmark(size_t rows, size_t inner, size_t cols,
                         size_t block_size, size_t iterations);
double threaded_blocked_benchmark(size_t rows, size_t inner, size_t cols,
                                  size_t num_threads, size_t block_size,
                                  size_t iterations);
void run_benchmark(size_t rows, size_t inner, size_t cols, size_t num_threads,
                   size_t block_size, size_t iterations);

#endif // BENCH_H
