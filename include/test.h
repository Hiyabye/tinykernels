#ifndef TEST_H
#define TEST_H

#include <stddef.h>

// Benchmarking function to test matrix multiplication performance
void run_benchmark(size_t rows, size_t inner, size_t cols, size_t num_threads,
                   size_t iterations);

#endif // TEST_H
