#include "test.h"
#include "matmul.h"
#include "matrix.h"

#include <stddef.h>
#include <stdio.h>

#define COLOR_RED "\x1b[31m"
#define COLOR_GREEN "\x1b[32m"
#define COLOR_RESET "\x1b[0m"
#define CHECK_MARK "\u2713"
#define CROSS_MARK "\u2718"
#define LABEL_SIZE 64

static int matrix_equal(const Matrix *a, const Matrix *b, double eps) {
  if (!a || !b || !a->data || !b->data) {
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

static int check_config(const Matrix *a, const Matrix *b, const Matrix *reference, MatmulConfig cfg) {
  char label[LABEL_SIZE];
  if (!matmul_config_label(cfg, label, sizeof(label))) {
    snprintf(label, sizeof(label), "unknown");
  }

  Matrix c = matmul(a, b, cfg);
  if (!c.data) {
    fprintf(stderr, "%s%s %s failed to produce output%s\n", COLOR_RED, CROSS_MARK, label, COLOR_RESET);
    return 0;
  }

  int ok = matrix_equal(reference, &c, 1e-6);
  matrix_free(&c);

  if (!ok) {
    fprintf(stderr, "%s%s %s result does not match reference%s\n", COLOR_RED, CROSS_MARK, label, COLOR_RESET);
    return 0;
  }

  return 1;
}

static int test_matmul_case(size_t rows, size_t inner, size_t cols, size_t threads, size_t block_size) {
  printf("\n[test] A=%zux%zu, B=%zux%zu, threads=%zu, block=%zu\n", rows, inner, inner, cols, threads, block_size);

  Matrix a = matrix_new(rows, inner);
  Matrix b = matrix_new(inner, cols);
  if (!a.data || !b.data) {
    matrix_free(&a);
    matrix_free(&b);
    return 0;
  }

  matrix_fill_pattern(&a);
  matrix_fill_pattern(&b);

  MatmulConfig reference_cfg = matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IJK, 0, 1, 1);
  Matrix reference = matmul(&a, &b, reference_cfg);
  if (!reference.data) {
    matrix_free(&a);
    matrix_free(&b);
    return 0;
  }

  MatmulConfig configs[] = {
      matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IJK, 0, 1, 1),
      matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IKJ, 0, 1, 1),
      matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IJK, 1, 1, block_size),
      matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IKJ, 1, 1, block_size),
      matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IJK, 0, threads, 1),
      matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IKJ, 0, threads, 1),
      matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IJK, 1, threads, block_size),
      matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IKJ, 1, threads, block_size),
#if TK_ENABLE_OPENMP
      matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IJK, 0, threads, 1),
      matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IKJ, 0, threads, 1),
      matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IJK, 1, threads, block_size),
      matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IKJ, 1, threads, block_size),
#endif
  };

  int all_passed = 1;
  for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); ++i) {
    all_passed &= check_config(&a, &b, &reference, configs[i]);
  }

  if (all_passed) {
    printf("%s%s all configurations match reference%s\n", COLOR_GREEN, CHECK_MARK, COLOR_RESET);
  }

  matrix_free(&a);
  matrix_free(&b);
  matrix_free(&reference);
  return all_passed;
}

int test_matmul_correctness(void) {
  int all_passed = 1;

  all_passed &= test_matmul_case(1, 1, 1, 4, 8);
  all_passed &= test_matmul_case(2, 3, 4, 4, 8);
  all_passed &= test_matmul_case(7, 11, 5, 4, 8);
  all_passed &= test_matmul_case(10, 10, 10, 4, 8);
  all_passed &= test_matmul_case(16, 16, 16, 4, 8);
  all_passed &= test_matmul_case(17, 17, 17, 4, 8);
  all_passed &= test_matmul_case(31, 29, 13, 4, 8);

  all_passed &= test_matmul_case(64, 64, 64, 1, 8);
  all_passed &= test_matmul_case(64, 64, 64, 2, 8);
  all_passed &= test_matmul_case(64, 64, 64, 4, 8);
  all_passed &= test_matmul_case(64, 64, 64, 8, 8);

  all_passed &= test_matmul_case(64, 64, 64, 4, 1);
  all_passed &= test_matmul_case(64, 64, 64, 4, 4);
  all_passed &= test_matmul_case(64, 64, 64, 4, 8);
  all_passed &= test_matmul_case(64, 64, 64, 4, 16);
  all_passed &= test_matmul_case(64, 64, 64, 4, 32);

  all_passed &= test_matmul_case(70, 70, 70, 4, 8);
  all_passed &= test_matmul_case(70, 70, 70, 4, 16);
  all_passed &= test_matmul_case(70, 70, 70, 4, 32);

  all_passed &= test_matmul_case(70, 70, 70, 3, 8);
  all_passed &= test_matmul_case(70, 70, 70, 5, 8);
  all_passed &= test_matmul_case(70, 70, 70, 6, 8);

  all_passed &= test_matmul_case(16, 16, 16, 4, 32);
  all_passed &= test_matmul_case(16, 16, 16, 4, 64);
  all_passed &= test_matmul_case(16, 16, 16, 4, 128);

  all_passed &= test_matmul_case(16, 32, 64, 4, 8);
  all_passed &= test_matmul_case(32, 16, 64, 4, 8);
  all_passed &= test_matmul_case(64, 16, 32, 4, 8);

  return all_passed;
}
