#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

// Matrix structure definition
typedef struct Matrix {
  size_t rows;
  size_t cols;
  int *data;
} Matrix;

// Matrix utility functions
Matrix empty_matrix(void);
Matrix init_matrix(size_t r, size_t c);
void print_matrix(const Matrix *mat);
void free_matrix(Matrix *mat);

// Matrix multiplication functions
Matrix matmul_naive(const Matrix *a, const Matrix *b);
Matrix matmul_threaded(const Matrix *a, const Matrix *b, size_t num_threads);
Matrix matmul_ikj(const Matrix *a, const Matrix *b);
Matrix matmul_blocked(const Matrix *a, const Matrix *b, size_t block_size);
Matrix matmul_threaded_blocked(const Matrix *a, const Matrix *b,
                               size_t num_threads, size_t block_size);

#endif // MATRIX_H
