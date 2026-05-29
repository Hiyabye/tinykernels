#include "kernels.h"

int tk_matmul_single_into(const Matrix *a, const Matrix *b, Matrix *c, MatmulConfig cfg) {
  if (cfg.use_blocking) {
    if (cfg.loop_order == MATMUL_LOOP_IJK) {
      tk_matmul_range_blocked_ijk(a, b, c, 0, a->rows, cfg.block_size);
    } else {
      tk_matmul_range_blocked_ikj(a, b, c, 0, a->rows, cfg.block_size);
    }
    return 1;
  }

  if (cfg.loop_order == MATMUL_LOOP_IJK) {
    tk_matmul_range_ijk(a, b, c, 0, a->rows);
  } else {
    tk_matmul_range_ikj(a, b, c, 0, a->rows);
  }

  return 1;
}
