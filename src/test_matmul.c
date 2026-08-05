#include "test_matmul.h"
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
#define CONFIG_CAPACITY 24

static void add_config(MatmulConfig *configs, size_t *count, MatmulConfig config) {
  if (*count < CONFIG_CAPACITY) {
    configs[*count] = config;
    *count += 1;
  }
}

static int matrix_equal(const Matrix *expected, const Matrix *actual, mat_elem_t tolerance) {
  if (!expected || !actual || !expected->data || !actual->data) return 0;

  if (expected->rows != actual->rows || expected->cols != actual->cols) return 0;

  for (size_t row = 0; row < expected->rows; ++row) {
    for (size_t col = 0; col < expected->cols; ++col) {
      mat_elem_t diff = expected->data[row * expected->cols + col] - actual->data[row * actual->cols + col];
      if (diff < -tolerance || diff > tolerance) return 0;
    }
  }

  return 1;
}

static int check_config(const Matrix *lhs, const Matrix *rhs, const Matrix *reference, MatmulConfig config) {
  char label[LABEL_SIZE];
  if (!matmul_config_label(config, label, sizeof(label))) snprintf(label, sizeof(label), "unknown");

  Matrix actual = matmul(lhs, rhs, config);
  if (!actual.data) {
    fprintf(stderr, "%s%s %s failed to produce output%s\n", COLOR_RED, CROSS_MARK, label, COLOR_RESET);
    return 0;
  }

  int matched = matrix_equal(reference, &actual, 1e-6);
  matrix_free(&actual);

  if (!matched) {
    fprintf(stderr, "%s%s %s result does not match reference%s\n", COLOR_RED, CROSS_MARK, label, COLOR_RESET);
    return 0;
  }

  return 1;
}

static int test_matmul_case(size_t rows, size_t inner, size_t cols, size_t threads, size_t block_size) {
  printf("\n[test] A=%zux%zu, B=%zux%zu, threads=%zu, block=%zu\n", rows, inner, inner, cols, threads, block_size);

  Matrix lhs = matrix_new(rows, inner);
  Matrix rhs = matrix_new(inner, cols);
  if (!lhs.data || !rhs.data) {
    matrix_free(&lhs);
    matrix_free(&rhs);
    return 0;
  }

  matrix_fill_pattern(&lhs);
  matrix_fill_pattern(&rhs);

  MatmulConfig ref_config = matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IJK, 0, 0, 1, 1);
  Matrix reference = matmul(&lhs, &rhs, ref_config);
  if (!reference.data) {
    matrix_free(&lhs);
    matrix_free(&rhs);
    return 0;
  }

  MatmulConfig configs[CONFIG_CAPACITY];
  size_t config_count = 0;

  add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IJK, 0, 0, 1, 1));
  add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IKJ, 0, 0, 1, 1));
  add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IJK, 1, 0, 1, block_size));
  add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IKJ, 1, 0, 1, block_size));
  add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IJK, 0, 0, threads, 1));
  add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IKJ, 0, 0, threads, 1));
  add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IJK, 1, 0, threads, block_size));
  add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IKJ, 1, 0, threads, block_size));

  if (matmul_simd_available()) {
    add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IKJ, 0, 1, 1, 1));
    add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IKJ, 1, 1, 1, block_size));
    add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IKJ, 0, 1, threads, 1));
    add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IKJ, 1, 1, threads, block_size));
  }

#if ENABLE_OPENMP
  add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IJK, 0, 0, threads, 1));
  add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IKJ, 0, 0, threads, 1));
  add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IJK, 1, 0, threads, block_size));
  add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IKJ, 1, 0, threads, block_size));

  if (matmul_simd_available()) {
    add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IKJ, 0, 1, threads, 1));
    add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IKJ, 1, 1, threads, block_size));
  }
#endif

  int all_passed = 1;
  for (size_t config_idx = 0; config_idx < config_count; ++config_idx) { all_passed &= check_config(&lhs, &rhs, &reference, configs[config_idx]); }

  if (all_passed) printf("%s%s all configurations match reference%s\n", COLOR_GREEN, CHECK_MARK, COLOR_RESET);

  matrix_free(&lhs);
  matrix_free(&rhs);
  matrix_free(&reference);
  return all_passed;
}

static int test_invalid_configs(void) {
  Matrix lhs = matrix_new(2, 3);
  Matrix rhs = matrix_new(3, 2);
  Matrix out = matrix_new(2, 2);

  if (!lhs.data || !rhs.data || !out.data) {
    matrix_free(&lhs);
    matrix_free(&rhs);
    matrix_free(&out);
    return 0;
  }

  matrix_fill_pattern(&lhs);
  matrix_fill_pattern(&rhs);

  int passed = 1;
  passed &= !matmul_into(&lhs, &rhs, &out, matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IKJ, 0, 0, 0, 1));
  passed &= !matmul_into(&lhs, &rhs, &out, matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IKJ, 1, 0, 1, 0));
  passed &= !matmul_into(&lhs, &rhs, &out, matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IJK, 0, 1, 1, 1));

  if (!matmul_simd_available()) { passed &= !matmul_into(&lhs, &rhs, &out, matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IKJ, 0, 1, 1, 1)); }

  matrix_free(&lhs);
  matrix_free(&rhs);
  matrix_free(&out);
  return passed;
}

int test_matmul_correctness(void) {
  int all_passed = 1;

  all_passed &= test_invalid_configs();

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
