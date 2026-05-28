#ifndef MATMUL_H
#define MATMUL_H

#include "matrix.h"

typedef enum {
  MATMUL_REF_IJK,
  MATMUL_SEQ_IKJ,
  MATMUL_SEQ_BLOCKED_IKJ,
  MATMUL_PAR_ROWS_IJK,
  MATMUL_PAR_ROWS_BLOCKED_IKJ,
  MATMUL_OPENMP_IKJ,
} MatmulKernel;

typedef struct {
  MatmulKernel kernel;
  size_t num_threads;
  size_t block_size;
} MatmulConfig;

Matrix matmul(const Matrix *a, const Matrix *b, MatmulConfig cfg);
const char *matmul_kernel_name(MatmulKernel kernel);

#endif // MATMUL_H
