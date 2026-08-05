#ifndef MATMUL_KERNELS_BACKENDS_H
#define MATMUL_KERNELS_BACKENDS_H

#include "matmul.h"

int tk_matmul_single_into(const Matrix *lhs, const Matrix *rhs, Matrix *out, MatmulConfig config);
int tk_matmul_pthread_into(const Matrix *lhs, const Matrix *rhs, Matrix *out, MatmulConfig config);
int tk_matmul_openmp_into(const Matrix *lhs, const Matrix *rhs, Matrix *out, MatmulConfig config);

#endif // MATMUL_KERNELS_BACKENDS_H
