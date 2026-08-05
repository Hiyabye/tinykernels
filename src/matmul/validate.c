#include "matmul.h"
#include "matrix.h"

#include <stdio.h>

int validate_matrices(const Matrix *lhs, const Matrix *rhs, const Matrix *out) {
  if (!lhs || !rhs || !out || !lhs->data || !rhs->data || !out->data) {
    fprintf(stderr, "invalid matrix\n");
    return 0;
  }

  if (lhs->rows == 0 || lhs->cols == 0 || rhs->rows == 0 || rhs->cols == 0 || out->rows == 0 || out->cols == 0) {
    fprintf(stderr, "matrix dimensions must be greater than zero\n");
    return 0;
  }

  if (lhs->cols != rhs->rows) {
    fprintf(stderr, "incompatible input matrix dimensions\n");
    return 0;
  }

  if (out->rows != lhs->rows || out->cols != rhs->cols) {
    fprintf(stderr, "invalid output matrix dimensions\n");
    return 0;
  }

  return 1;
}

int validate_input_matrices(const Matrix *lhs, const Matrix *rhs) {
  if (!lhs || !rhs || !lhs->data || !rhs->data) {
    fprintf(stderr, "invalid matrix multiplication request\n");
    return 0;
  }

  if (lhs->rows == 0 || lhs->cols == 0 || rhs->rows == 0 || rhs->cols == 0) {
    fprintf(stderr, "matrix dimensions must be greater than zero\n");
    return 0;
  }

  if (lhs->cols != rhs->rows) {
    fprintf(stderr, "incompatible input matrix dimensions\n");
    return 0;
  }

  return 1;
}

int validate_config(MatmulConfig config) {
  if (config.backend != MATMUL_BACKEND_SINGLE && config.backend != MATMUL_BACKEND_PTHREAD &&
      config.backend != MATMUL_BACKEND_OPENMP) {
    fprintf(stderr, "invalid matmul backend\n");
    return 0;
  }

  if (config.loop_order != MATMUL_LOOP_IJK && config.loop_order != MATMUL_LOOP_IKJ) {
    fprintf(stderr, "invalid matmul loop order\n");
    return 0;
  }

  if (config.backend != MATMUL_BACKEND_SINGLE && config.num_threads == 0) {
    fprintf(stderr, "num_threads must be greater than zero\n");
    return 0;
  }

  if (config.use_blocking && config.block_size == 0) {
    fprintf(stderr, "block_size must be greater than zero\n");
    return 0;
  }

  if (config.use_simd && !matmul_simd_available()) {
    fprintf(stderr, "SIMD support is unavailable on this target\n");
    return 0;
  }

  if (config.use_simd && config.loop_order != MATMUL_LOOP_IKJ) {
    fprintf(stderr, "SIMD currently requires IKJ loop order\n");
    return 0;
  }

  return 1;
}
