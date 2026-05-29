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

  size_t len = rows * cols;
  if (len > SIZE_MAX / sizeof(mat_elem_t)) {
    fprintf(stderr, "matrix byte size overflow\n");
    return (Matrix){0, 0, NULL};
  }

  Matrix m = {rows, cols, calloc(len, sizeof(mat_elem_t))};
  if (!m.data) {
    fprintf(stderr, "memory allocation failed\n");
    return (Matrix){0, 0, NULL};
  }

  return m;
}

void matrix_free(Matrix *m) {
  if (!m || !m->data) {
    return;
  }

  free(m->data);
  m->data = NULL;
  m->rows = 0;
  m->cols = 0;
}

void matrix_fill(Matrix *m, mat_elem_t value) {
  if (!m || !m->data) {
    return;
  }

  for (size_t i = 0; i < m->rows * m->cols; ++i) {
    m->data[i] = value;
  }
}

void matrix_fill_pattern(Matrix *m) {
  if (!m || !m->data) {
    return;
  }

  for (size_t i = 0; i < m->rows; ++i) {
    for (size_t j = 0; j < m->cols; ++j) {
      m->data[i * m->cols + j] = (mat_elem_t)((i + j) % 10 + 1);
    }
  }
}

void matrix_print(const Matrix *m) {
  if (!m || !m->data) {
    fprintf(stderr, "invalid matrix\n");
    return;
  }

  for (size_t i = 0; i < m->rows; ++i) {
    for (size_t j = 0; j < m->cols; ++j) {
      printf("%f ", m->data[i * m->cols + j]);
    }
    printf("\n");
  }
}
