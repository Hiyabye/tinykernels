#include "tk_common.h"
#include "tk_kernels.h"

#if ENABLE_OPENMP
#include <omp.h>
#endif

tk_status tk_backend_openmp(const TkMatrix *lhs, const TkMatrix *rhs, TkMatrix *out, const tk_matmul_cfg *cfg) {
#if ENABLE_OPENMP
  if (cfg->use_blocking) {
#pragma omp parallel for num_threads(cfg->num_threads) schedule(static)
    for (size_t row0 = 0; row0 < lhs->rows; row0 += cfg->block_size) {
      size_t row1 = tk_min(row0 + cfg->block_size, lhs->rows);
      tk_kernel_run(lhs, rhs, out, cfg, row0, row1);
    }
    return TK_OK;
  }

#pragma omp parallel for num_threads(cfg->num_threads) schedule(static)
  for (size_t row = 0; row < lhs->rows; ++row) tk_kernel_run(lhs, rhs, out, cfg, row, row + 1);

  return TK_OK;
#else
  (void)lhs;
  (void)rhs;
  (void)out;
  (void)cfg;
  TK_LOGE("OpenMP support is disabled");
  return TK_ERR_UNSUPPORTED;
#endif
}
