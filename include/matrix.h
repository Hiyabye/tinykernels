#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

// Matrix structure definition
struct Matrix {
  size_t rows;
  size_t cols;
  int **data;
};

// Matrix utility functions
struct Matrix empty_matrix(void);
struct Matrix init_matrix(size_t r, size_t c);
void print_matrix(const struct Matrix *mat);
void free_matrix(struct Matrix *mat);

// Matrix multiplication functions
struct Matrix matmul_naive(const struct Matrix *a, const struct Matrix *b);
struct Matrix matmul_threaded(const struct Matrix *a, const struct Matrix *b,
                              size_t num_threads);

#endif // MATRIX_H
