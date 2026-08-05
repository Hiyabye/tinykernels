#ifndef MATMUL_KERNELS_DISPATCH_H
#define MATMUL_KERNELS_DISPATCH_H

#include "loop.h"
#include "simd.h"

static inline void tk_matmul_range_scalar(const Matrix *lhs, const Matrix *rhs, Matrix *out, MatmulConfig config,
                                          size_t row_start, size_t row_end) {
  if (config.use_blocking) {
    if (config.loop_order == MATMUL_LOOP_IJK) {
      tk_matmul_range_blocked_ijk(lhs, rhs, out, row_start, row_end, config.block_size);
    } else {
      tk_matmul_range_blocked_ikj(lhs, rhs, out, row_start, row_end, config.block_size);
    }
    return;
  }

  if (config.loop_order == MATMUL_LOOP_IJK) {
    tk_matmul_range_ijk(lhs, rhs, out, row_start, row_end);
  } else {
    tk_matmul_range_ikj(lhs, rhs, out, row_start, row_end);
  }
}

static inline void tk_matmul_range(const Matrix *lhs, const Matrix *rhs, Matrix *out, MatmulConfig config,
                                   size_t row_start, size_t row_end) {
  if (config.use_simd && TK_HAVE_SIMD) {
    if (config.use_blocking) {
      tk_matmul_range_blocked_simd_ikj(lhs, rhs, out, row_start, row_end, config.block_size);
    } else {
      tk_matmul_range_simd_ikj(lhs, rhs, out, row_start, row_end);
    }
    return;
  }

  tk_matmul_range_scalar(lhs, rhs, out, config, row_start, row_end);
}

#endif // MATMUL_KERNELS_DISPATCH_H
