#include "tk_kernels.h"

tk_status tk_backend_single(const TkMatrix *lhs, const TkMatrix *rhs, TkMatrix *out, const tk_matmul_cfg *cfg) {
  tk_kernel_run(lhs, rhs, out, cfg, 0, lhs->rows);
  return TK_OK;
}
