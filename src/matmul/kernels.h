#ifndef TINYKERNELS_MATMUL_KERNELS_H
#define TINYKERNELS_MATMUL_KERNELS_H

#include "matmul.h"

#include <stddef.h>

static inline size_t tk_min_size(size_t a, size_t b) { return a < b ? a : b; }

static inline void tk_matmul_range_ijk(const Matrix *a, const Matrix *b, Matrix *c, size_t row_start, size_t row_end) {
  for (size_t i = row_start; i < row_end; ++i) {
    for (size_t j = 0; j < b->cols; ++j) {
      mat_elem_t sum = 0;
      for (size_t k = 0; k < a->cols; ++k) {
        sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
      }
      c->data[i * c->cols + j] = sum;
    }
  }
}

static inline void tk_matmul_range_ikj(const Matrix *a, const Matrix *b, Matrix *c, size_t row_start, size_t row_end) {
  for (size_t i = row_start; i < row_end; ++i) {
    for (size_t k = 0; k < a->cols; ++k) {
      mat_elem_t aik = a->data[i * a->cols + k];
      for (size_t j = 0; j < b->cols; ++j) {
        c->data[i * c->cols + j] += aik * b->data[k * b->cols + j];
      }
    }
  }
}

static inline void tk_matmul_range_blocked_ijk(const Matrix *a, const Matrix *b, Matrix *c, size_t row_start,
                                               size_t row_end, size_t block_size) {
  for (size_t i0 = row_start; i0 < row_end; i0 += block_size) {
    size_t i_max = tk_min_size(i0 + block_size, row_end);

    for (size_t j0 = 0; j0 < b->cols; j0 += block_size) {
      size_t j_max = tk_min_size(j0 + block_size, b->cols);

      for (size_t k0 = 0; k0 < a->cols; k0 += block_size) {
        size_t k_max = tk_min_size(k0 + block_size, a->cols);

        for (size_t i = i0; i < i_max; ++i) {
          for (size_t j = j0; j < j_max; ++j) {
            mat_elem_t sum = c->data[i * c->cols + j];
            for (size_t k = k0; k < k_max; ++k) {
              sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
            c->data[i * c->cols + j] = sum;
          }
        }
      }
    }
  }
}

static inline void tk_matmul_range_blocked_ikj(const Matrix *a, const Matrix *b, Matrix *c, size_t row_start,
                                               size_t row_end, size_t block_size) {
  for (size_t i0 = row_start; i0 < row_end; i0 += block_size) {
    size_t i_max = tk_min_size(i0 + block_size, row_end);

    for (size_t k0 = 0; k0 < a->cols; k0 += block_size) {
      size_t k_max = tk_min_size(k0 + block_size, a->cols);

      for (size_t j0 = 0; j0 < b->cols; j0 += block_size) {
        size_t j_max = tk_min_size(j0 + block_size, b->cols);

        for (size_t i = i0; i < i_max; ++i) {
          for (size_t k = k0; k < k_max; ++k) {
            mat_elem_t aik = a->data[i * a->cols + k];
            for (size_t j = j0; j < j_max; ++j) {
              c->data[i * c->cols + j] += aik * b->data[k * b->cols + j];
            }
          }
        }
      }
    }
  }
}

static inline void tk_matmul_range(const Matrix *a, const Matrix *b, Matrix *c, MatmulConfig cfg, size_t row_start,
                                   size_t row_end) {
  if (cfg.use_blocking) {
    if (cfg.loop_order == MATMUL_LOOP_IJK) {
      tk_matmul_range_blocked_ijk(a, b, c, row_start, row_end, cfg.block_size);
    } else {
      tk_matmul_range_blocked_ikj(a, b, c, row_start, row_end, cfg.block_size);
    }
    return;
  }

  if (cfg.loop_order == MATMUL_LOOP_IJK) {
    tk_matmul_range_ijk(a, b, c, row_start, row_end);
  } else {
    tk_matmul_range_ikj(a, b, c, row_start, row_end);
  }
}

int tk_matmul_single_into(const Matrix *a, const Matrix *b, Matrix *c, MatmulConfig cfg);
int tk_matmul_pthread_into(const Matrix *a, const Matrix *b, Matrix *c, MatmulConfig cfg);
int tk_matmul_openmp_into(const Matrix *a, const Matrix *b, Matrix *c, MatmulConfig cfg);

#endif // TINYKERNELS_MATMUL_KERNELS_H
