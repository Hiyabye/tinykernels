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

  MatmulConfig cfg_seq_ikj = {
      .kernel = MATMUL_SEQ_IKJ,
      .num_threads = 1,
      .block_size = 1,
  };
  Matrix c_seq_ikj = matmul(&a, &b, cfg_seq_ikj);

  MatmulConfig cfg_seq_blocked_ikj = {
      .kernel = MATMUL_SEQ_BLOCKED_IKJ,
      .num_threads = 1,
      .block_size = 16,
  };
  Matrix c_seq_blocked_ikj = matmul(&a, &b, cfg_seq_blocked_ikj);

  MatmulConfig cfg_par_rows_ijk = {
      .kernel = MATMUL_PAR_ROWS_IJK,
      .num_threads = 4,
      .block_size = 1,
  };
  Matrix c_par_rows_ijk = matmul(&a, &b, cfg_par_rows_ijk);

  MatmulConfig cfg_par_rows_blocked_ikj = {
      .kernel = MATMUL_PAR_ROWS_BLOCKED_IKJ,
      .num_threads = 4,
      .block_size = 16,
  };
  Matrix c_par_rows_blocked_ikj = matmul(&a, &b, cfg_par_rows_blocked_ikj);

  MatmulConfig cfg_openmp_ikj = {
      .kernel = MATMUL_OPENMP_IKJ,
      .num_threads = 4,
      .block_size = 1,
  };
  Matrix c_openmp_ikj = matmul(&a, &b, cfg_openmp_ikj);

  if (!c_ref.data || !c_seq_ikj.data || !c_seq_blocked_ikj.data || !c_par_rows_ijk.data ||
      !c_par_rows_blocked_ikj.data || !c_openmp_ikj.data) {
    fprintf(stderr, "test matrix multiplication failed\n");
    matrix_free(&a);
    matrix_free(&b);
    matrix_free(&c_ref);
    matrix_free(&c_seq_ikj);
    matrix_free(&c_seq_blocked_ikj);
    matrix_free(&c_par_rows_ijk);
    matrix_free(&c_par_rows_blocked_ikj);
    matrix_free(&c_openmp_ikj);
    return;
  }

  if (!matrix_equal(&c_ref, &c_seq_ikj, 1e-6)) {
    fprintf(stderr, "ikj result does not match reference\n");
  } else {
    printf("%s%s ikj result matches reference%s\n", COLOR_GREEN, CHECK_MARK, COLOR_RESET);
  }

  if (!matrix_equal(&c_ref, &c_seq_blocked_ikj, 1e-6)) {
    fprintf(stderr, "blocked result does not match reference\n");
  } else {
    printf("%s%s blocked result matches reference%s\n", COLOR_GREEN, CHECK_MARK, COLOR_RESET);
  }

  if (!matrix_equal(&c_ref, &c_par_rows_ijk, 1e-6)) {
    fprintf(stderr, "threaded result does not match reference\n");
  } else {
    printf("%s%s threaded result matches reference%s\n", COLOR_GREEN, CHECK_MARK, COLOR_RESET);
  }

  if (!matrix_equal(&c_ref, &c_par_rows_blocked_ikj, 1e-6)) {
    fprintf(stderr, "threaded blocked result does not match reference\n");
  } else {
    printf("%s%s threaded blocked result matches reference%s\n", COLOR_GREEN, CHECK_MARK, COLOR_RESET);
  }

  if (!matrix_equal(&c_ref, &c_openmp_ikj, 1e-6)) {
    fprintf(stderr, "OpenMP result does not match reference\n");
  } else {
    printf("%s%s OpenMP result matches reference%s\n", COLOR_GREEN, CHECK_MARK, COLOR_RESET);
  }

  matrix_free(&a);
  matrix_free(&b);
  matrix_free(&c_ref);
  matrix_free(&c_seq_ikj);
  matrix_free(&c_seq_blocked_ikj);
  matrix_free(&c_par_rows_ijk);
  matrix_free(&c_par_rows_blocked_ikj);
  matrix_free(&c_openmp_ikj);
}
