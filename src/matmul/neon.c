#include "kernels.h"
#include "matrix.h"

#include <assert.h>
#include <stddef.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

void tk_matmul_range_simd_ikj(const Matrix *a, const Matrix *b, Matrix *c, size_t row_start, size_t row_end) {
#if defined(__aarch64__)
  if (sizeof(mat_elem_t) == sizeof(float)) {
    for (size_t i = row_start; i < row_end; ++i) {
      for (size_t k = 0; k < a->cols; ++k) {
        float32x4_t aik_vec = vdupq_n_f32(a->data[i * a->cols + k]);
        size_t j = 0;

        for (; j + 4 <= b->cols; j += 4) {
          float32x4_t b_vec = vld1q_f32(&b->data[k * b->cols + j]);
          float32x4_t c_vec = vld1q_f32(&c->data[i * c->cols + j]);

          c_vec = vfmaq_f32(c_vec, aik_vec, b_vec);
          vst1q_f32(&c->data[i * c->cols + j], c_vec);
        }

        for (; j < b->cols; ++j) {
          c->data[i * c->cols + j] += a->data[i * a->cols + k] * b->data[k * b->cols + j];
        }
      }
    }
  } else if (sizeof(mat_elem_t) == sizeof(double)) {
    assert(0 && "not implemented yet");
  }
#else
  (void)a;
  (void)b;
  (void)c;
  (void)row_start;
  (void)row_end;
#endif
}

void tk_matmul_range_blocked_simd_ikj(const Matrix *a, const Matrix *b, Matrix *c, size_t row_start, size_t row_end,
                                      size_t block_size) {
#if defined(__aarch64__)
  if (sizeof(mat_elem_t) == sizeof(float)) {
    for (size_t i0 = row_start; i0 < row_end; i0 += block_size) {
      size_t i_max = tk_min_size(i0 + block_size, row_end);

      for (size_t k0 = 0; k0 < a->cols; k0 += block_size) {
        size_t k_max = tk_min_size(k0 + block_size, a->cols);

        for (size_t j0 = 0; j0 < b->cols; j0 += block_size) {
          size_t j_max = tk_min_size(j0 + block_size, b->cols);

          for (size_t i = i0; i < i_max; ++i) {
            for (size_t k = k0; k < k_max; ++k) {
              float32x4_t aik_vec = vdupq_n_f32(a->data[i * a->cols + k]);
              size_t j = j0;

              for (; j + 4 <= j_max; j += 4) {
                float32x4_t b_vec = vld1q_f32(&b->data[k * b->cols + j]);
                float32x4_t c_vec = vld1q_f32(&c->data[i * c->cols + j]);

                c_vec = vfmaq_f32(c_vec, aik_vec, b_vec);
                vst1q_f32(&c->data[i * c->cols + j], c_vec);
              }

              for (; j < j_max; ++j) {
                c->data[i * c->cols + j] += a->data[i * a->cols + k] * b->data[k * b->cols + j];
              }
            }
          }
        }
      }
    }
  } else if (sizeof(mat_elem_t) == sizeof(double)) {
  }
#else
  (void)a;
  (void)b;
  (void)c;
  (void)row_start;
  (void)row_end;
  (void)block_size;
#endif
}
