#include "matrix.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Returns an empty matrix (0 rows, 0 cols, NULL data)
Matrix empty_matrix(void) {
  Matrix m = {0, 0, NULL};
  return m;
}

// Initializes a matrix with given rows and columns, allocates memory for data
Matrix init_matrix(size_t r, size_t c) {
  if (r == 0 || c == 0) {
    fprintf(stderr, "invalid matrix dimensions\n");
    return empty_matrix();
  }

  if (r > SIZE_MAX / c) {
    fprintf(stderr, "matrix size overflow\n");
    return empty_matrix();
  }

  Matrix m = {r, c, NULL};
  m.data = calloc(r * c, sizeof(mat_elem_t));
  if (!m.data) {
    fprintf(stderr, "memory allocation failed\n");
    return empty_matrix();
  }
  return m;
}

// Prints the contents of a matrix to stdout
void print_matrix(const Matrix *m) {
  for (size_t i = 0; i < m->rows; ++i) {
    for (size_t j = 0; j < m->cols; ++j) {
      printf("%f ", m->data[i * m->cols + j]);
    }
    printf("\n");
  }
}

// Frees the memory allocated for a matrix and resets its fields
void free_matrix(Matrix *m) {
  if (!m || !m->data) {
    return;
  }
  free(m->data);
  m->data = NULL;
  m->rows = 0;
  m->cols = 0;
}
