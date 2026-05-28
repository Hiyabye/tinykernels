#include "test.h"
#include "matmul.h"
#include "matrix.h"

#include <stdio.h>

static void test_matrix_fill(Matrix *m) {
  for (size_t i = 0; i < m->rows; ++i) {
    for (size_t j = 0; j < m->cols; ++j) {
      m->data[i * m->cols + j] = (mat_elem_t)((i + j) % 10 + 1);
    }
  }
}

int matrix_equal(const Matrix *a, const Matrix *b, double eps) {
  if (!a || !b || !a->data || !b->data) {
    fprintf(stderr, "invalid matrix\n");
    return 0;
  }

  if (a->rows != b->rows || a->cols != b->cols) {
    return 0;
  }

  for (size_t i = 0; i < a->rows; ++i) {
    for (size_t j = 0; j < a->cols; ++j) {
      double diff = a->data[i * a->cols + j] - b->data[i * b->cols + j];
      if (diff < -eps || diff > eps) {
        return 0;
      }
    }
  }

  return 1;
}

void test_matmul_correctness(void) {
  size_t rows = 10;
  size_t inner = 10;
  size_t cols = 10;

  Matrix a = matrix_new(rows, inner);
  Matrix b = matrix_new(inner, cols);

  if (!a.data || !b.data) {
    fprintf(stderr, "test matrix allocation failed\n");
    matrix_free(&a);
    matrix_free(&b);
    return;
  }

  test_matrix_fill(&a);
  test_matrix_fill(&b);

  MatmulConfig cfg_ref = {
      .kernel = MATMUL_REF_IJK,
      .num_threads = 1,
      .block_size = 1,
  };
  Matrix c_ref = matmul(&a, &b, cfg_ref);

  MatmulConfig cfg_ikj = {
      .kernel = MATMUL_SEQ_IKJ,
      .num_threads = 1,
      .block_size = 1,
  };
  Matrix c_ikj = matmul(&a, &b, cfg_ikj);

  MatmulConfig cfg_blocked = {
      .kernel = MATMUL_SEQ_BLOCKED_IKJ,
      .num_threads = 1,
      .block_size = 4,
  };
  Matrix c_blocked = matmul(&a, &b, cfg_blocked);

  MatmulConfig cfg_threaded = {
      .kernel = MATMUL_PAR_ROWS_IJK,
      .num_threads = 4,
      .block_size = 1,
  };
  Matrix c_threaded = matmul(&a, &b, cfg_threaded);

  MatmulConfig cfg_threaded_blocked = {
      .kernel = MATMUL_PAR_ROWS_BLOCKED_IKJ,
      .num_threads = 4,
      .block_size = 4,
  };
  Matrix c_threaded_blocked = matmul(&a, &b, cfg_threaded_blocked);

  if (!c_ref.data || !c_ikj.data || !c_blocked.data || !c_threaded.data ||
      !c_threaded_blocked.data) {
    fprintf(stderr, "test matrix multiplication failed\n");
    matrix_free(&a);
    matrix_free(&b);
    matrix_free(&c_ref);
    matrix_free(&c_ikj);
    matrix_free(&c_blocked);
    matrix_free(&c_threaded);
    matrix_free(&c_threaded_blocked);
    return;
  }

  if (!matrix_equal(&c_ref, &c_ikj, 1e-6)) {
    fprintf(stderr, "ikj result does not match reference\n");
  } else {
    printf("%s%s ikj result matches reference%s\n", COLOR_GREEN, CHECK_MARK,
           COLOR_RESET);
  }

  if (!matrix_equal(&c_ref, &c_blocked, 1e-6)) {
    fprintf(stderr, "blocked result does not match reference\n");
  } else {
    printf("%s%s blocked result matches reference%s\n", COLOR_GREEN, CHECK_MARK,
           COLOR_RESET);
  }

  if (!matrix_equal(&c_ref, &c_threaded, 1e-6)) {
    fprintf(stderr, "threaded result does not match reference\n");
  } else {
    printf("%s%s threaded result matches reference%s\n", COLOR_GREEN,
           CHECK_MARK, COLOR_RESET);
  }

  if (!matrix_equal(&c_ref, &c_threaded_blocked, 1e-6)) {
    fprintf(stderr, "threaded blocked result does not match reference\n");
  } else {
    printf("%s%s threaded blocked result matches reference%s\n", COLOR_GREEN,
           CHECK_MARK, COLOR_RESET);
  }

  matrix_free(&a);
  matrix_free(&b);
  matrix_free(&c_ref);
  matrix_free(&c_ikj);
  matrix_free(&c_blocked);
  matrix_free(&c_threaded);
  matrix_free(&c_threaded_blocked);
}
