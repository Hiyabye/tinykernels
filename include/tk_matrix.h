#ifndef TK_MATRIX_H
#define TK_MATRIX_H

#include <stddef.h>

typedef float tk_elem_t;

typedef struct {
  size_t rows;
  size_t cols;
  tk_elem_t *data;
} TkMatrix;

/* Allocates via tk_xcalloc (aborts on OOM). Returns the zero sentinel and logs
 * only for zero dimensions or size overflow; otherwise allocation never fails. */
TkMatrix tk_mat_new(size_t rows, size_t cols);
void tk_mat_free(TkMatrix *mat);
void tk_mat_fill(TkMatrix *mat, tk_elem_t value);
void tk_mat_fill_pattern(TkMatrix *mat);
void tk_mat_print(const TkMatrix *mat);

#endif /* TK_MATRIX_H */
