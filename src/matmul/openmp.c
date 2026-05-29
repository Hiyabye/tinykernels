#include "kernels.h"

#if TK_ENABLE_OPENMP
#include <omp.h>
#endif

#include <stdio.h>

int tk_matmul_openmp_into(const Matrix *a, const Matrix *b, Matrix *c, MatmulConfig cfg) {
#if TK_ENABLE_OPENMP
  if (cfg.use_blocking) {
#pragma omp parallel for num_threads(cfg.num_threads) schedule(static)
    for (size_t i0 = 0; i0 < a->rows; i0 += cfg.block_size) {
      size_t row_end = tk_min_size(i0 + cfg.block_size, a->rows);
      tk_matmul_range(a, b, c, cfg, i0, row_end);
    }
    return 1;
  }

#pragma omp parallel for num_threads(cfg.num_threads) schedule(static)
  for (size_t i = 0; i < a->rows; ++i) {
    tk_matmul_range(a, b, c, cfg, i, i + 1);
  }

  return 1;
#else
  (void)a;
  (void)b;
  (void)c;
  (void)cfg;
  fprintf(stderr, "OpenMP support is disabled\n");
  return 0;
#endif
}
