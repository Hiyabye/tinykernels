#ifndef MATMUL_H
#define MATMUL_H

#include "matrix.h"

// Matrix multiplication functions
Matrix matmul_naive(const Matrix *a, const Matrix *b);
Matrix matmul_threaded(const Matrix *a, const Matrix *b, size_t num_threads);
Matrix matmul_ikj(const Matrix *a, const Matrix *b);
Matrix matmul_blocked(const Matrix *a, const Matrix *b, size_t block_size);
Matrix matmul_threaded_blocked(const Matrix *a, const Matrix *b,
                               size_t num_threads, size_t block_size);

#endif // MATMUL_H
