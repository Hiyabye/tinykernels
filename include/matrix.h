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
void matrix_free(Matrix *m);
void matrix_fill(Matrix *m, mat_elem_t value);
void matrix_fill_pattern(Matrix *m);
void matrix_print(const Matrix *m);

#endif // MATRIX_H
