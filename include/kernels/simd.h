#ifndef MATMUL_KERNELS_SIMD_H
#define MATMUL_KERNELS_SIMD_H

#include "matmul.h"

#include <stddef.h>

// implemented in simd.c
void matmul_range_simd_ikj(const Matrix *lhs, const Matrix *rhs, Matrix *out, size_t row_start, size_t row_end);
void matmul_range_blocked_simd_ikj(const Matrix *lhs, const Matrix *rhs, Matrix *out, size_t row_start, size_t row_end, size_t block_size);

#endif // MATMUL_KERNELS_SIMD_H
