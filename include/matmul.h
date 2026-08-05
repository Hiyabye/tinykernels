#ifndef MATMUL_H
#define MATMUL_H

#include "matrix.h"

#include <stdbool.h>
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
  bool use_blocking;
  bool use_simd;
  size_t num_threads;
  size_t block_size;
} MatmulConfig;

MatmulConfig matmul_config(MatmulBackend backend, MatmulLoopOrder loop_order, bool use_blocking, bool use_simd, size_t num_threads,
                           size_t block_size);

int matmul_into(const Matrix *lhs, const Matrix *rhs, Matrix *out, MatmulConfig config);
Matrix matmul(const Matrix *lhs, const Matrix *rhs, MatmulConfig config);

const char *matmul_backend_name(MatmulBackend backend);
const char *matmul_loop_order_name(MatmulLoopOrder loop_order);
int matmul_simd_available(void);
int matmul_config_label(MatmulConfig config, char *label, size_t label_size);

#endif // MATMUL_H
