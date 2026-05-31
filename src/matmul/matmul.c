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

static int validate_matrices(const Matrix *a, const Matrix *b, const Matrix *c) {
  if (!a || !b || !c || !a->data || !b->data || !c->data) {
    fprintf(stderr, "invalid matrix\n");
    return 0;
  }

  if (a->cols != b->rows) {
    fprintf(stderr, "incompatible input matrix dimensions\n");
    return 0;
  }

  if (c->rows != a->rows || c->cols != b->cols) {
    fprintf(stderr, "invalid output matrix dimensions\n");
    return 0;
  }

  return 1;
}

static int validate_config(MatmulConfig cfg) {
  if (cfg.backend != MATMUL_BACKEND_SINGLE && cfg.backend != MATMUL_BACKEND_PTHREAD &&
      cfg.backend != MATMUL_BACKEND_OPENMP) {
    fprintf(stderr, "invalid matmul backend\n");
    return 0;
  }

  if (cfg.loop_order != MATMUL_LOOP_IJK && cfg.loop_order != MATMUL_LOOP_IKJ) {
    fprintf(stderr, "invalid matmul loop order\n");
    return 0;
  }

  if (cfg.backend != MATMUL_BACKEND_SINGLE && cfg.num_threads == 0) {
    fprintf(stderr, "num_threads must be greater than zero\n");
    return 0;
  }

  if (cfg.use_blocking && cfg.block_size == 0) {
    fprintf(stderr, "block_size must be greater than zero\n");
    return 0;
  }

  if (cfg.use_simd && cfg.loop_order != MATMUL_LOOP_IKJ) {
    fprintf(stderr, "SIMD currently requires IKJ loop order\n");
    return 0;
  }

  return 1;
}

int matmul_into(const Matrix *a, const Matrix *b, Matrix *c, MatmulConfig cfg) {
  if (!validate_matrices(a, b, c) || !validate_config(cfg)) {
    return 0;
  }

  matrix_fill(c, 0.0);

  switch (cfg.backend) {
  case MATMUL_BACKEND_SINGLE:
    return tk_matmul_single_into(a, b, c, cfg);
  case MATMUL_BACKEND_PTHREAD:
    return tk_matmul_pthread_into(a, b, c, cfg);
  case MATMUL_BACKEND_OPENMP:
    return tk_matmul_openmp_into(a, b, c, cfg);
  default:
    return 0;
  }
}

Matrix matmul(const Matrix *a, const Matrix *b, MatmulConfig cfg) {
  if (!a || !b || !a->data || !b->data || a->cols != b->rows) {
    fprintf(stderr, "invalid matrix multiplication request\n");
    return (Matrix){0, 0, NULL};
  }

  Matrix c = matrix_new(a->rows, b->cols);
  if (!c.data) {
    return (Matrix){0, 0, NULL};
  }

  if (!matmul_into(a, b, &c, cfg)) {
    matrix_free(&c);
    return (Matrix){0, 0, NULL};
  }

  return c;
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

int matmul_config_label(MatmulConfig cfg, char *buf, size_t buf_size) {
  if (!buf || buf_size == 0) {
    return 0;
  }

  const char *backend = matmul_backend_name(cfg.backend);
  const char *loop = matmul_loop_order_name(cfg.loop_order);
  const char *blocking = cfg.use_blocking ? "blocked" : "plain";
  const char *simd = cfg.use_simd ? "simd" : "sisd";

  int written = snprintf(buf, buf_size, "%s_%s_%s_%s", backend, blocking, simd, loop);
  return written >= 0 && (size_t)written < buf_size;
}
