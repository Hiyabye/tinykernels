#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

typedef float mat_elem_t;

typedef struct {
  size_t rows;
  size_t cols;
  mat_elem_t *data;
} Matrix;

Matrix matrix_new(size_t rows, size_t cols);
void matrix_free(Matrix *mat);
void matrix_fill(Matrix *mat, mat_elem_t value);
void matrix_fill_pattern(Matrix *mat);
void matrix_print(const Matrix *mat);

#endif // MATRIX_H
