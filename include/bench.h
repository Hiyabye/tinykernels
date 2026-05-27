#ifndef BENCH_H
#define BENCH_H

#include <stddef.h>

void bench_run_case(size_t rows, size_t inner, size_t cols, size_t threads,
                    size_t block_size, size_t iterations);
void bench_run_default_suite(void);

#endif // BENCH_H
