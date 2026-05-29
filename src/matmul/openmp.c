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
      if (cfg.loop_order == MATMUL_LOOP_IJK) {
        tk_matmul_range_blocked_ijk(a, b, c, i0, row_end, cfg.block_size);
      } else {
        tk_matmul_range_blocked_ikj(a, b, c, i0, row_end, cfg.block_size);
      }
    }
    return 1;
  }

#pragma omp parallel for num_threads(cfg.num_threads) schedule(static)
  for (size_t i = 0; i < a->rows; ++i) {
    if (cfg.loop_order == MATMUL_LOOP_IJK) {
      tk_matmul_range_ijk(a, b, c, i, i + 1);
    } else {
      tk_matmul_range_ikj(a, b, c, i, i + 1);
    }
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
