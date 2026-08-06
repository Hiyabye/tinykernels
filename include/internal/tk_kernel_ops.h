#ifndef TK_KERNEL_OPS_H
#define TK_KERNEL_OPS_H

#include "tk_backend.h"

#include <stddef.h>

/* Loop kernels: static inline scalar/loop-order variants. SIMD kernels are
 * declared here and defined in src/kernels/tk_simd.c. */

static inline void tk_kernel_ijk(const TkMatrix *lhs, const TkMatrix *rhs, TkMatrix *out, size_t row_start, size_t row_end) {
  for (size_t row = row_start; row < row_end; ++row) {
    for (size_t col = 0; col < rhs->cols; ++col) {
      tk_elem_t sum = 0;
      for (size_t inner = 0; inner < lhs->cols; ++inner) { sum += lhs->data[row * lhs->cols + inner] * rhs->data[inner * rhs->cols + col]; }
      out->data[row * out->cols + col] = sum;
    }
  }
}

static inline void tk_kernel_ikj(const TkMatrix *lhs, const TkMatrix *rhs, TkMatrix *out, size_t row_start, size_t row_end) {
  for (size_t row = row_start; row < row_end; ++row) {
    for (size_t inner = 0; inner < lhs->cols; ++inner) {
      tk_elem_t lhs_val = lhs->data[row * lhs->cols + inner];
      for (size_t col = 0; col < rhs->cols; ++col) { out->data[row * out->cols + col] += lhs_val * rhs->data[inner * rhs->cols + col]; }
    }
  }
}

static inline void tk_kernel_ijk_blocked(const TkMatrix *lhs, const TkMatrix *rhs, TkMatrix *out, size_t row_start, size_t row_end,
                                         size_t block_size) {
  for (size_t row0 = row_start; row0 < row_end; row0 += block_size) {
    size_t row1 = tk_min(row0 + block_size, row_end);

    for (size_t col0 = 0; col0 < rhs->cols; col0 += block_size) {
      size_t col1 = tk_min(col0 + block_size, rhs->cols);

      for (size_t inner0 = 0; inner0 < lhs->cols; inner0 += block_size) {
        size_t inner1 = tk_min(inner0 + block_size, lhs->cols);

        for (size_t row = row0; row < row1; ++row) {
          for (size_t col = col0; col < col1; ++col) {
            tk_elem_t sum = out->data[row * out->cols + col];
            for (size_t inner = inner0; inner < inner1; ++inner) { sum += lhs->data[row * lhs->cols + inner] * rhs->data[inner * rhs->cols + col]; }
            out->data[row * out->cols + col] = sum;
          }
        }
      }
    }
  }
}

static inline void tk_kernel_ikj_blocked(const TkMatrix *lhs, const TkMatrix *rhs, TkMatrix *out, size_t row_start, size_t row_end,
                                         size_t block_size) {
  for (size_t row0 = row_start; row0 < row_end; row0 += block_size) {
    size_t row1 = tk_min(row0 + block_size, row_end);

    for (size_t inner0 = 0; inner0 < lhs->cols; inner0 += block_size) {
      size_t inner1 = tk_min(inner0 + block_size, lhs->cols);

      for (size_t col0 = 0; col0 < rhs->cols; col0 += block_size) {
        size_t col1 = tk_min(col0 + block_size, rhs->cols);

        for (size_t row = row0; row < row1; ++row) {
          for (size_t inner = inner0; inner < inner1; ++inner) {
            tk_elem_t lhs_val = lhs->data[row * lhs->cols + inner];
            for (size_t col = col0; col < col1; ++col) { out->data[row * out->cols + col] += lhs_val * rhs->data[inner * rhs->cols + col]; }
          }
        }
      }
    }
  }
}

void tk_kernel_ikj_simd(const TkMatrix *lhs, const TkMatrix *rhs, TkMatrix *out, size_t row_start, size_t row_end);
void tk_kernel_ikj_blocked_simd(const TkMatrix *lhs, const TkMatrix *rhs, TkMatrix *out, size_t row_start, size_t row_end, size_t block_size);

/* Scalar dispatcher: blocking + loop order selection. */
static inline void tk_kernel_run_scalar(const TkMatrix *lhs, const TkMatrix *rhs, TkMatrix *out, const tk_matmul_cfg *cfg, size_t row_start,
                                        size_t row_end) {
  if (cfg->use_blocking) {
    if (cfg->loop_order == TK_LOOP_IJK) {
      tk_kernel_ijk_blocked(lhs, rhs, out, row_start, row_end, cfg->block_size);
    } else {
      tk_kernel_ikj_blocked(lhs, rhs, out, row_start, row_end, cfg->block_size);
    }
    return;
  }

  if (cfg->loop_order == TK_LOOP_IJK) {
    tk_kernel_ijk(lhs, rhs, out, row_start, row_end);
  } else {
    tk_kernel_ikj(lhs, rhs, out, row_start, row_end);
  }
}

/* Single entry point backends call to compute rows [row0, row1). */
static inline void tk_kernel_run(const TkMatrix *lhs, const TkMatrix *rhs, TkMatrix *out, const tk_matmul_cfg *cfg, size_t row_start,
                                 size_t row_end) {
#if defined(__SSE__)
  if (cfg->use_simd) {
    if (cfg->use_blocking) {
      tk_kernel_ikj_blocked_simd(lhs, rhs, out, row_start, row_end, cfg->block_size);
    } else {
      tk_kernel_ikj_simd(lhs, rhs, out, row_start, row_end);
    }
    return;
  }
#endif

  tk_kernel_run_scalar(lhs, rhs, out, cfg, row_start, row_end);
}

#endif /* TK_KERNEL_OPS_H */
