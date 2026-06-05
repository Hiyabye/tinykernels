#include "matrix.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

Matrix matrix_new(size_t rows, size_t cols) {
  if (rows == 0 || cols == 0) {
    fprintf(stderr, "invalid matrix dimensions\n");
    return (Matrix){0, 0, NULL};
  }

  if (rows > SIZE_MAX / cols) {
    fprintf(stderr, "matrix size overflow\n");
    return (Matrix){0, 0, NULL};
  }

  size_t element_count = rows * cols;
  if (element_count > SIZE_MAX / sizeof(mat_elem_t)) {
    fprintf(stderr, "matrix byte size overflow\n");
    return (Matrix){0, 0, NULL};
  }

  Matrix mat = {rows, cols, calloc(element_count, sizeof(mat_elem_t))};
  if (!mat.data) {
    fprintf(stderr, "memory allocation failed\n");
    return (Matrix){0, 0, NULL};
  }

  return mat;
}

void matrix_free(Matrix *mat) {
  if (!mat || !mat->data) return;

  free(mat->data);
  mat->data = NULL;
  mat->rows = 0;
  mat->cols = 0;
}

void matrix_fill(Matrix *mat, mat_elem_t value) {
  if (!mat || !mat->data) return;

  size_t element_count = mat->rows * mat->cols;
  for (size_t index = 0; index < element_count; ++index) mat->data[index] = value;
}

void matrix_fill_pattern(Matrix *mat) {
  if (!mat || !mat->data) return;

  for (size_t row = 0; row < mat->rows; ++row) {
    for (size_t col = 0; col < mat->cols; ++col) {
      mat->data[row * mat->cols + col] = (mat_elem_t)((row + col) % 10 + 1);
    }
  }
}

void matrix_print(const Matrix *mat) {
  if (!mat || !mat->data) {
    fprintf(stderr, "invalid matrix\n");
    return;
  }

  for (size_t row = 0; row < mat->rows; ++row) {
    for (size_t col = 0; col < mat->cols; ++col) printf("%f ", mat->data[row * mat->cols + col]);
    printf("\n");
  }
}
