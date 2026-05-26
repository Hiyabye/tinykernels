#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

// Element type for matrix entries
typedef double mat_elem_t;

// Matrix structure definition
typedef struct Matrix {
  size_t rows;
  size_t cols;
  mat_elem_t *data;
} Matrix;

// Matrix utility functions
Matrix empty_matrix(void);
Matrix init_matrix(size_t r, size_t c);
void print_matrix(const Matrix *mat);
void free_matrix(Matrix *mat);

#endif // MATRIX_H
