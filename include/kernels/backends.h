#ifndef MATMUL_KERNELS_BACKENDS_H
#define MATMUL_KERNELS_BACKENDS_H

#include "matmul.h"

int matmul_single_into(const Matrix *lhs, const Matrix *rhs, Matrix *out, MatmulConfig config);
int matmul_pthread_into(const Matrix *lhs, const Matrix *rhs, Matrix *out, MatmulConfig config);
int matmul_openmp_into(const Matrix *lhs, const Matrix *rhs, Matrix *out, MatmulConfig config);
int validate_matrices(const Matrix *lhs, const Matrix *rhs, const Matrix *out);
int validate_input_matrices(const Matrix *lhs, const Matrix *rhs);
int validate_config(MatmulConfig config);

#endif // MATMUL_KERNELS_BACKENDS_H
