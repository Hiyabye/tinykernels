#include "test.h"
#include "matmul.h"
#include "matrix.h"

#include <stddef.h>
#include <stdio.h>

#define COLOR_RED "\x1b[31m"
#define COLOR_GREEN "\x1b[32m"
#define COLOR_YELLOW "\x1b[33m"
#define COLOR_BLUE "\x1b[34m"
#define COLOR_MAGENTA "\x1b[35m"
#define COLOR_CYAN "\x1b[36m"
#define COLOR_RESET "\x1b[0m"
#define CHECK_MARK "\u2713"
#define CROSS_MARK "\u2718"

static void test_matrix_fill(Matrix *m) {
  for (size_t i = 0; i < m->rows; ++i) {
    for (size_t j = 0; j < m->cols; ++j) {
      m->data[i * m->cols + j] = (mat_elem_t)((i + j) % 10 + 1);
    }
  }
}

static int matrix_equal(const Matrix *a, const Matrix *b, double eps) {
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

static int test_matmul_case(size_t rows, size_t inner, size_t cols, size_t threads, size_t block_size) {
  printf("\n[test] A: %zux%zu, B: %zux%zu, threads: %zu, block size: %zu\n", rows, inner, inner, cols, threads,
         block_size);

  Matrix a = matrix_new(rows, inner);
  Matrix b = matrix_new(inner, cols);

  if (!a.data || !b.data) {
    fprintf(stderr, "test matrix allocation failed\n");
    matrix_free(&a);
    matrix_free(&b);
    return 0;
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
      .block_size = block_size,
  };
  Matrix c_seq_blocked_ikj = matmul(&a, &b, cfg_seq_blocked_ikj);

  MatmulConfig cfg_par_rows_ijk = {
      .kernel = MATMUL_PAR_ROWS_IJK,
      .num_threads = threads,
      .block_size = 1,
  };
  Matrix c_par_rows_ijk = matmul(&a, &b, cfg_par_rows_ijk);

  MatmulConfig cfg_par_rows_blocked_ikj = {
      .kernel = MATMUL_PAR_ROWS_BLOCKED_IKJ,
      .num_threads = threads,
      .block_size = block_size,
  };
  Matrix c_par_rows_blocked_ikj = matmul(&a, &b, cfg_par_rows_blocked_ikj);

  MatmulConfig cfg_openmp_ikj = {
      .kernel = MATMUL_OPENMP_IKJ,
      .num_threads = threads,
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
    return 0;
  }

  int all_passed = 1;

  if (!matrix_equal(&c_ref, &c_seq_ikj, 1e-6)) {
    fprintf(stderr, "%s%s ikj result does not match reference%s\n", COLOR_RED, CROSS_MARK, COLOR_RESET);
    all_passed = 0;
  } else {
    printf("%s%s ikj result matches reference%s\n", COLOR_GREEN, CHECK_MARK, COLOR_RESET);
  }

  if (!matrix_equal(&c_ref, &c_seq_blocked_ikj, 1e-6)) {
    fprintf(stderr, "%s%s blocked ikj result does not match reference%s\n", COLOR_RED, CROSS_MARK, COLOR_RESET);
    all_passed = 0;
  } else {
    printf("%s%s blocked result matches reference%s\n", COLOR_GREEN, CHECK_MARK, COLOR_RESET);
  }

  if (!matrix_equal(&c_ref, &c_par_rows_ijk, 1e-6)) {
    fprintf(stderr, "%s%s threaded result does not match reference%s\n", COLOR_RED, CROSS_MARK, COLOR_RESET);
    all_passed = 0;
  } else {
    printf("%s%s threaded result matches reference%s\n", COLOR_GREEN, CHECK_MARK, COLOR_RESET);
  }

  if (!matrix_equal(&c_ref, &c_par_rows_blocked_ikj, 1e-6)) {
    fprintf(stderr, "%s%s threaded blocked result does not match reference%s\n", COLOR_RED, CROSS_MARK, COLOR_RESET);
    all_passed = 0;
  } else {
    printf("%s%s threaded blocked result matches reference%s\n", COLOR_GREEN, CHECK_MARK, COLOR_RESET);
  }

  if (!matrix_equal(&c_ref, &c_openmp_ikj, 1e-6)) {
    fprintf(stderr, "%s%s OpenMP result does not match reference%s\n", COLOR_RED, CROSS_MARK, COLOR_RESET);
    all_passed = 0;
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

  return all_passed;
}

int test_matmul_correctness(void) {
  int all_passed = 1;

  // matrix size sweep
  all_passed &= test_matmul_case(1, 1, 1, 4, 8);
  all_passed &= test_matmul_case(2, 3, 4, 4, 8);
  all_passed &= test_matmul_case(7, 11, 5, 4, 8);
  all_passed &= test_matmul_case(10, 10, 10, 4, 8);
  all_passed &= test_matmul_case(16, 16, 16, 4, 8);
  all_passed &= test_matmul_case(17, 17, 17, 4, 8);
  all_passed &= test_matmul_case(31, 29, 13, 4, 8);

  // thread count sweep
  all_passed &= test_matmul_case(64, 64, 64, 1, 8);
  all_passed &= test_matmul_case(64, 64, 64, 2, 8);
  all_passed &= test_matmul_case(64, 64, 64, 4, 8);
  all_passed &= test_matmul_case(64, 64, 64, 8, 8);

  // block size sweep
  all_passed &= test_matmul_case(64, 64, 64, 4, 1);
  all_passed &= test_matmul_case(64, 64, 64, 4, 4);
  all_passed &= test_matmul_case(64, 64, 64, 4, 8);
  all_passed &= test_matmul_case(64, 64, 64, 4, 16);
  all_passed &= test_matmul_case(64, 64, 64, 4, 32);

  // block_size is not a divisor of rows/cols
  all_passed &= test_matmul_case(70, 70, 70, 4, 8);
  all_passed &= test_matmul_case(70, 70, 70, 4, 16);
  all_passed &= test_matmul_case(70, 70, 70, 4, 32);

  // thread count is not a divisor of rows
  all_passed &= test_matmul_case(70, 70, 70, 3, 8);
  all_passed &= test_matmul_case(70, 70, 70, 5, 8);
  all_passed &= test_matmul_case(70, 70, 70, 6, 8);

  // block_size > rows
  all_passed &= test_matmul_case(16, 16, 16, 4, 32);
  all_passed &= test_matmul_case(16, 16, 16, 4, 64);
  all_passed &= test_matmul_case(16, 16, 16, 4, 128);

  // non-square matrix
  all_passed &= test_matmul_case(16, 32, 64, 4, 8);
  all_passed &= test_matmul_case(32, 16, 64, 4, 8);
  all_passed &= test_matmul_case(64, 16, 32, 4, 8);

  return all_passed;
}
