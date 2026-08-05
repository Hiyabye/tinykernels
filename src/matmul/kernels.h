#ifndef TINYKERNELS_MATMUL_KERNELS_H
#define TINYKERNELS_MATMUL_KERNELS_H

#include "matmul.h"

#include <stddef.h>

static inline size_t tk_min_size(size_t x, size_t y) { return x < y ? x : y; }

#define TK_HAVE_SIMD TK_HAVE_SSE

#if defined(__SSE__)
#define TK_HAVE_SSE 1
#else
#define TK_HAVE_SSE 0
#endif

static inline void tk_matmul_range_ijk(const Matrix *lhs, const Matrix *rhs, Matrix *out, size_t row_start,
                                       size_t row_end) {
  for (size_t row = row_start; row < row_end; ++row) {
    for (size_t col = 0; col < rhs->cols; ++col) {
      mat_elem_t sum = 0;
      for (size_t inner = 0; inner < lhs->cols; ++inner) {
        sum += lhs->data[row * lhs->cols + inner] * rhs->data[inner * rhs->cols + col];
      }
      out->data[row * out->cols + col] = sum;
    }
  }
}

static inline void tk_matmul_range_ikj(const Matrix *lhs, const Matrix *rhs, Matrix *out, size_t row_start,
                                       size_t row_end) {
  for (size_t row = row_start; row < row_end; ++row) {
    for (size_t inner = 0; inner < lhs->cols; ++inner) {
      mat_elem_t lhs_val = lhs->data[row * lhs->cols + inner];
      for (size_t col = 0; col < rhs->cols; ++col) {
        out->data[row * out->cols + col] += lhs_val * rhs->data[inner * rhs->cols + col];
      }
    }
  }
}

static inline void tk_matmul_range_blocked_ijk(const Matrix *lhs, const Matrix *rhs, Matrix *out, size_t row_start,
                                               size_t row_end, size_t block_size) {
  for (size_t row0 = row_start; row0 < row_end; row0 += block_size) {
    size_t row1 = tk_min_size(row0 + block_size, row_end);

    for (size_t col0 = 0; col0 < rhs->cols; col0 += block_size) {
      size_t col1 = tk_min_size(col0 + block_size, rhs->cols);

      for (size_t inner0 = 0; inner0 < lhs->cols; inner0 += block_size) {
        size_t inner1 = tk_min_size(inner0 + block_size, lhs->cols);

        for (size_t row = row0; row < row1; ++row) {
          for (size_t col = col0; col < col1; ++col) {
            mat_elem_t sum = out->data[row * out->cols + col];
            for (size_t inner = inner0; inner < inner1; ++inner) {
              sum += lhs->data[row * lhs->cols + inner] * rhs->data[inner * rhs->cols + col];
            }
            out->data[row * out->cols + col] = sum;
          }
        }
      }
    }
  }
}

static inline void tk_matmul_range_blocked_ikj(const Matrix *lhs, const Matrix *rhs, Matrix *out, size_t row_start,
                                               size_t row_end, size_t block_size) {
  for (size_t row0 = row_start; row0 < row_end; row0 += block_size) {
    size_t row1 = tk_min_size(row0 + block_size, row_end);

    for (size_t inner0 = 0; inner0 < lhs->cols; inner0 += block_size) {
      size_t inner1 = tk_min_size(inner0 + block_size, lhs->cols);

      for (size_t col0 = 0; col0 < rhs->cols; col0 += block_size) {
        size_t col1 = tk_min_size(col0 + block_size, rhs->cols);

        for (size_t row = row0; row < row1; ++row) {
          for (size_t inner = inner0; inner < inner1; ++inner) {
            mat_elem_t lhs_val = lhs->data[row * lhs->cols + inner];
            for (size_t col = col0; col < col1; ++col) {
              out->data[row * out->cols + col] += lhs_val * rhs->data[inner * rhs->cols + col];
            }
          }
        }
      }
    }
  }
}

// implemented in simd.c
void tk_matmul_range_simd_ikj(const Matrix *lhs, const Matrix *rhs, Matrix *out, size_t row_start, size_t row_end);
void tk_matmul_range_blocked_simd_ikj(const Matrix *lhs, const Matrix *rhs, Matrix *out, size_t row_start,
                                      size_t row_end, size_t block_size);

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

int tk_matmul_single_into(const Matrix *lhs, const Matrix *rhs, Matrix *out, MatmulConfig config);
int tk_matmul_pthread_into(const Matrix *lhs, const Matrix *rhs, Matrix *out, MatmulConfig config);
int tk_matmul_openmp_into(const Matrix *lhs, const Matrix *rhs, Matrix *out, MatmulConfig config);

#endif // TINYKERNELS_MATMUL_KERNELS_H
