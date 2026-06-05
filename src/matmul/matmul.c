#include "matmul.h"
#include "kernels.h"
#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>

MatmulConfig matmul_config(MatmulBackend backend, MatmulLoopOrder loop_order, int use_blocking, int use_simd,
                           size_t num_threads, size_t block_size) {
  return (MatmulConfig){
      .backend = backend,
      .loop_order = loop_order,
      .use_blocking = use_blocking,
      .use_simd = use_simd,
      .num_threads = num_threads,
      .block_size = block_size,
  };
}

static int validate_matrices(const Matrix *lhs, const Matrix *rhs, const Matrix *out) {
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

static int validate_input_matrices(const Matrix *lhs, const Matrix *rhs) {
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

static int validate_config(MatmulConfig config) {
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

int matmul_into(const Matrix *lhs, const Matrix *rhs, Matrix *out, MatmulConfig config) {
  if (!validate_matrices(lhs, rhs, out) || !validate_config(config)) {
    return 0;
  }

  matrix_fill(out, 0.0);

  switch (config.backend) {
  case MATMUL_BACKEND_SINGLE:
    return tk_matmul_single_into(lhs, rhs, out, config);
  case MATMUL_BACKEND_PTHREAD:
    return tk_matmul_pthread_into(lhs, rhs, out, config);
  case MATMUL_BACKEND_OPENMP:
    return tk_matmul_openmp_into(lhs, rhs, out, config);
  default:
    return 0;
  }
}

Matrix matmul(const Matrix *lhs, const Matrix *rhs, MatmulConfig config) {
  if (!validate_input_matrices(lhs, rhs) || !validate_config(config)) {
    return (Matrix){0, 0, NULL};
  }

  Matrix out = matrix_new(lhs->rows, rhs->cols);
  if (!out.data) {
    return (Matrix){0, 0, NULL};
  }

  if (!matmul_into(lhs, rhs, &out, config)) {
    matrix_free(&out);
    return (Matrix){0, 0, NULL};
  }

  return out;
}

const char *matmul_backend_name(MatmulBackend backend) {
  switch (backend) {
  case MATMUL_BACKEND_SINGLE:
    return "single";
  case MATMUL_BACKEND_PTHREAD:
    return "pthread";
  case MATMUL_BACKEND_OPENMP:
    return "openmp";
  default:
    return "unknown";
  }
}

const char *matmul_loop_order_name(MatmulLoopOrder loop_order) {
  switch (loop_order) {
  case MATMUL_LOOP_IJK:
    return "ijk";
  case MATMUL_LOOP_IKJ:
    return "ikj";
  default:
    return "unknown";
  }
}

int matmul_simd_available(void) { return sizeof(mat_elem_t) == sizeof(float) && TK_HAVE_SIMD; }

int matmul_config_label(MatmulConfig config, char *label, size_t label_size) {
  if (!label || label_size == 0) {
    return 0;
  }

  const char *backend_name = matmul_backend_name(config.backend);
  const char *loop_order_name = matmul_loop_order_name(config.loop_order);
  const char *blocking_name = config.use_blocking ? "blocked" : "plain";
  const char *simd_name = config.use_simd ? "simd" : "sisd";

  int written = snprintf(label, label_size, "%s_%s_%s_%s", backend_name, blocking_name, simd_name, loop_order_name);
  return written >= 0 && (size_t)written < label_size;
}
