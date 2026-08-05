#ifndef MATMUL_KERNELS_SIMD_H
#define MATMUL_KERNELS_SIMD_H

#include "matmul.h"

#include <stddef.h>

#define TK_HAVE_SIMD TK_HAVE_SSE

#if defined(__SSE__)
#define TK_HAVE_SSE 1
#else
#define TK_HAVE_SSE 0
#endif

// implemented in simd.c
void tk_matmul_range_simd_ikj(const Matrix *lhs, const Matrix *rhs, Matrix *out, size_t row_start, size_t row_end);
void tk_matmul_range_blocked_simd_ikj(const Matrix *lhs, const Matrix *rhs, Matrix *out, size_t row_start,
                                      size_t row_end, size_t block_size);

#endif // MATMUL_KERNELS_SIMD_H
