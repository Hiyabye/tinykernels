#ifndef MATMUL_H
#define MATMUL_H

#include "matrix.h"

#include <stddef.h>

typedef enum {
  MATMUL_BACKEND_SINGLE,
  MATMUL_BACKEND_PTHREAD,
  MATMUL_BACKEND_OPENMP,
} MatmulBackend;

typedef enum {
  MATMUL_LOOP_IJK,
  MATMUL_LOOP_IKJ,
} MatmulLoopOrder;

typedef struct {
  MatmulBackend backend;
  MatmulLoopOrder loop_order;
  int use_blocking;
  size_t num_threads;
  size_t block_size;
} MatmulConfig;

int matmul_into(const Matrix *a, const Matrix *b, Matrix *c, MatmulConfig cfg);
Matrix matmul(const Matrix *a, const Matrix *b, MatmulConfig cfg);

const char *matmul_backend_name(MatmulBackend backend);
const char *matmul_loop_order_name(MatmulLoopOrder loop_order);
int matmul_config_label(MatmulConfig cfg, char *buf, size_t buf_size);

#endif // MATMUL_H
