#include "kernels.h"

#if ENABLE_OPENMP
#include <omp.h>
#endif

#include <stdio.h>

int matmul_openmp_into(const Matrix *lhs, const Matrix *rhs, Matrix *out, MatmulConfig config) {
#if ENABLE_OPENMP
  if (config.use_blocking) {
#pragma omp parallel for num_threads(config.num_threads) schedule(static)
    for (size_t row0 = 0; row0 < lhs->rows; row0 += config.block_size) {
      size_t row1 = min_size(row0 + config.block_size, lhs->rows);
      matmul_range(lhs, rhs, out, config, row0, row1);
    }
    return 1;
  }

#pragma omp parallel for num_threads(config.num_threads) schedule(static)
  for (size_t row = 0; row < lhs->rows; ++row) matmul_range(lhs, rhs, out, config, row, row + 1);

  return 1;
#else
  (void)lhs;
  (void)rhs;
  (void)out;
  (void)config;
  fprintf(stderr, "OpenMP support is disabled\n");
  return 0;
#endif
}
