#include "kernels.h"
#include "matrix.h"

#if defined(__SSE__)
#include <xmmintrin.h>
#endif

#include <stddef.h>

void matmul_range_simd_ikj(const Matrix *lhs, const Matrix *rhs, Matrix *out, size_t row_start, size_t row_end) {
#if defined(__SSE__)

  for (size_t row = row_start; row < row_end; ++row) {
    for (size_t inner = 0; inner < lhs->cols; ++inner) {
      __m128 lhs_vec = _mm_set1_ps(lhs->data[row * lhs->cols + inner]);
      size_t col = 0;

      for (; col + 4 <= rhs->cols; col += 4) {
        __m128 rhs_vec = _mm_loadu_ps(&rhs->data[inner * rhs->cols + col]);
        __m128 out_vec = _mm_loadu_ps(&out->data[row * out->cols + col]);

        out_vec = _mm_add_ps(out_vec, _mm_mul_ps(lhs_vec, rhs_vec));
        _mm_storeu_ps(&out->data[row * out->cols + col], out_vec);
      }

      for (; col < rhs->cols; ++col) { out->data[row * out->cols + col] += lhs->data[row * lhs->cols + inner] * rhs->data[inner * rhs->cols + col]; }
    }
  }
#else
  (void)lhs;
  (void)rhs;
  (void)out;
  (void)row_start;
  (void)row_end;
#endif
}

void matmul_range_blocked_simd_ikj(const Matrix *lhs, const Matrix *rhs, Matrix *out, size_t row_start, size_t row_end, size_t block_size) {
#if defined(__SSE__)

  for (size_t row0 = row_start; row0 < row_end; row0 += block_size) {
    size_t row1 = min_size(row0 + block_size, row_end);

    for (size_t inner0 = 0; inner0 < lhs->cols; inner0 += block_size) {
      size_t inner1 = min_size(inner0 + block_size, lhs->cols);

      for (size_t col0 = 0; col0 < rhs->cols; col0 += block_size) {
        size_t col1 = min_size(col0 + block_size, rhs->cols);

        for (size_t row = row0; row < row1; ++row) {
          for (size_t inner = inner0; inner < inner1; ++inner) {
            __m128 lhs_vec = _mm_set1_ps(lhs->data[row * lhs->cols + inner]);
            size_t col = col0;

            for (; col + 4 <= col1; col += 4) {
              __m128 rhs_vec = _mm_loadu_ps(&rhs->data[inner * rhs->cols + col]);
              __m128 out_vec = _mm_loadu_ps(&out->data[row * out->cols + col]);

              out_vec = _mm_add_ps(out_vec, _mm_mul_ps(lhs_vec, rhs_vec));
              _mm_storeu_ps(&out->data[row * out->cols + col], out_vec);
            }

            for (; col < col1; ++col) { out->data[row * out->cols + col] += lhs->data[row * lhs->cols + inner] * rhs->data[inner * rhs->cols + col]; }
          }
        }
      }
    }
  }
#else
  (void)lhs;
  (void)rhs;
  (void)out;
  (void)row_start;
  (void)row_end;
  (void)block_size;
#endif
}
