#include "kernels.h"

int tk_matmul_single_into(const Matrix *a, const Matrix *b, Matrix *c, MatmulConfig cfg) {
  tk_matmul_range(a, b, c, cfg, 0, a->rows);
  return 1;
}
