#include "kernels.h"

int tk_matmul_single_into(const Matrix *lhs, const Matrix *rhs, Matrix *out, MatmulConfig config) {
  tk_matmul_range(lhs, rhs, out, config, 0, lhs->rows);
  return 1;
}
