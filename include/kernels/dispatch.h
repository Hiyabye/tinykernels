#ifndef MATMUL_KERNELS_DISPATCH_H
#define MATMUL_KERNELS_DISPATCH_H

#include "loop.h"
#include "simd.h"

static inline void matmul_range_scalar(const Matrix *lhs, const Matrix *rhs, Matrix *out, MatmulConfig config,
                                       size_t row_start, size_t row_end) {
  if (config.use_blocking) {
    if (config.loop_order == MATMUL_LOOP_IJK) {
      matmul_range_blocked_ijk(lhs, rhs, out, row_start, row_end, config.block_size);
    } else {
      matmul_range_blocked_ikj(lhs, rhs, out, row_start, row_end, config.block_size);
    }
    return;
  }

  if (config.loop_order == MATMUL_LOOP_IJK) {
    matmul_range_ijk(lhs, rhs, out, row_start, row_end);
  } else {
    matmul_range_ikj(lhs, rhs, out, row_start, row_end);
  }
}

static inline void matmul_range(const Matrix *lhs, const Matrix *rhs, Matrix *out, MatmulConfig config,
                                size_t row_start, size_t row_end) {
#if defined(__SSE__)
  if (config.use_simd) {
    if (config.use_blocking) {
      matmul_range_blocked_simd_ikj(lhs, rhs, out, row_start, row_end, config.block_size);
    } else {
      matmul_range_simd_ikj(lhs, rhs, out, row_start, row_end);
    }
    return;
  }
#endif

  matmul_range_scalar(lhs, rhs, out, config, row_start, row_end);
}

#endif // MATMUL_KERNELS_DISPATCH_H
