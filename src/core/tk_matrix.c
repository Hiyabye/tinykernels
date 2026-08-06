#include "tk_matrix.h"
#include "tk_common.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

TkMatrix tk_mat_new(size_t rows, size_t cols) {
  if (rows == 0 || cols == 0) {
    TK_LOGE("invalid matrix dimensions");
    return (TkMatrix){0, 0, NULL};
  }

  if (rows > SIZE_MAX / cols) {
    TK_LOGE("matrix size overflow");
    return (TkMatrix){0, 0, NULL};
  }

  size_t element_count = rows * cols;
  if (element_count > SIZE_MAX / sizeof(tk_elem_t)) {
    TK_LOGE("matrix byte size overflow");
    return (TkMatrix){0, 0, NULL};
  }

  return (TkMatrix){rows, cols, tk_xcalloc(element_count, sizeof(tk_elem_t))};
}

void tk_mat_free(TkMatrix *mat) {
  if (!mat || !mat->data) return;

  free(mat->data);
  mat->data = NULL;
  mat->rows = 0;
  mat->cols = 0;
}

void tk_mat_fill(TkMatrix *mat, tk_elem_t value) {
  if (!mat || !mat->data) return;

  size_t element_count = mat->rows * mat->cols;
  for (size_t index = 0; index < element_count; ++index) mat->data[index] = value;
}

void tk_mat_fill_pattern(TkMatrix *mat) {
  if (!mat || !mat->data) return;

  for (size_t row = 0; row < mat->rows; ++row) {
    for (size_t col = 0; col < mat->cols; ++col) { mat->data[row * mat->cols + col] = (tk_elem_t)((row + col) % 10 + 1); }
  }
}

void tk_mat_print(const TkMatrix *mat) {
  if (!mat || !mat->data) {
    fprintf(stderr, "invalid matrix\n");
    return;
  }

  for (size_t row = 0; row < mat->rows; ++row) {
    for (size_t col = 0; col < mat->cols; ++col) fprintf(stderr, "%f ", mat->data[row * mat->cols + col]);
    fputc('\n', stderr);
  }
}
