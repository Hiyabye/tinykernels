#include "tk_test.h"
#include "tk_backend.h"
#include "tk_gguf.h"
#include "tk_matrix.h"
#include "tk_model.h"
#include "tk_tokenizer.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COLOR_RED "\x1b[31m"
#define COLOR_GREEN "\x1b[32m"
#define COLOR_RESET "\x1b[0m"
#define CHECK_MARK "\u2713"
#define CROSS_MARK "\u2718"
#define LABEL_SIZE 64
#define CONFIG_CAPACITY 24

static unsigned g_failures = 0;

#define CHECK(cond)                                                                                                                                  \
  do {                                                                                                                                               \
    if (!(cond)) {                                                                                                                                   \
      fprintf(stderr, "%s%s FAILED: %s (line %d)%s\n", COLOR_RED, CROSS_MARK, #cond, __LINE__, COLOR_RESET);                                         \
      g_failures++;                                                                                                                                  \
    }                                                                                                                                                \
  } while (0)

static void add_config(tk_matmul_cfg *configs, size_t *count, tk_matmul_cfg config) {
  if (*count < CONFIG_CAPACITY) {
    configs[*count] = config;
    *count += 1;
  }
}

static int matrix_equal(const TkMatrix *expected, const TkMatrix *actual, tk_elem_t tolerance) {
  if (!expected || !actual || !expected->data || !actual->data) return 0;

  if (expected->rows != actual->rows || expected->cols != actual->cols) return 0;

  for (size_t row = 0; row < expected->rows; ++row) {
    for (size_t col = 0; col < expected->cols; ++col) {
      tk_elem_t diff = expected->data[row * expected->cols + col] - actual->data[row * actual->cols + col];
      if (diff < -tolerance || diff > tolerance) return 0;
    }
  }

  return 1;
}

static int check_config(const TkMatrix *lhs, const TkMatrix *rhs, const TkMatrix *reference, const tk_matmul_cfg *cfg) {
  char label[LABEL_SIZE];
  if (!tk_matmul_cfg_label(cfg, label, sizeof(label))) snprintf(label, sizeof(label), "unknown");

  tk_status status;
  TkMatrix actual = tk_mat_mul(lhs, rhs, cfg, &status);
  if (status != TK_OK || !actual.data) {
    fprintf(stderr, "%s%s %s failed to produce output%s\n", COLOR_RED, CROSS_MARK, label, COLOR_RESET);
    g_failures++;
    return 0;
  }

  int matched = matrix_equal(reference, &actual, 1e-6);
  tk_mat_free(&actual);

  if (!matched) {
    fprintf(stderr, "%s%s %s result does not match reference%s\n", COLOR_RED, CROSS_MARK, label, COLOR_RESET);
    g_failures++;
    return 0;
  }

  return 1;
}

static void test_matmul_case(size_t rows, size_t inner, size_t cols, size_t threads, size_t block_size) {
  printf("\n[test] A=%zux%zu, B=%zux%zu, threads=%zu, block=%zu\n", rows, inner, inner, cols, threads, block_size);

  TkMatrix lhs = tk_mat_new(rows, inner);
  TkMatrix rhs = tk_mat_new(inner, cols);

  tk_mat_fill_pattern(&lhs);
  tk_mat_fill_pattern(&rhs);

  tk_matmul_cfg ref_cfg = tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IJK, 0, 0, 1, 1);
  tk_status status;
  TkMatrix reference = tk_mat_mul(&lhs, &rhs, &ref_cfg, &status);
  if (status != TK_OK || !reference.data) {
    tk_mat_free(&lhs);
    tk_mat_free(&rhs);
    g_failures++;
    return;
  }

  tk_matmul_cfg configs[CONFIG_CAPACITY];
  size_t config_count = 0;

  add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IJK, 0, 0, 1, 1));
  add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IKJ, 0, 0, 1, 1));
  add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IJK, 1, 0, 1, block_size));
  add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IKJ, 1, 0, 1, block_size));
  add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_PTHREAD, TK_LOOP_IJK, 0, 0, threads, 1));
  add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_PTHREAD, TK_LOOP_IKJ, 0, 0, threads, 1));
  add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_PTHREAD, TK_LOOP_IJK, 1, 0, threads, block_size));
  add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_PTHREAD, TK_LOOP_IKJ, 1, 0, threads, block_size));

  if (tk_sse_available()) {
    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IKJ, 0, 1, 1, 1));
    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IKJ, 1, 1, 1, block_size));
    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_PTHREAD, TK_LOOP_IKJ, 0, 1, threads, 1));
    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_PTHREAD, TK_LOOP_IKJ, 1, 1, threads, block_size));
  }

#if ENABLE_OPENMP
  add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_OPENMP, TK_LOOP_IJK, 0, 0, threads, 1));
  add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_OPENMP, TK_LOOP_IKJ, 0, 0, threads, 1));
  add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_OPENMP, TK_LOOP_IJK, 1, 0, threads, block_size));
  add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_OPENMP, TK_LOOP_IKJ, 1, 0, threads, block_size));

  if (tk_sse_available()) {
    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_OPENMP, TK_LOOP_IKJ, 0, 1, threads, 1));
    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_OPENMP, TK_LOOP_IKJ, 1, 1, threads, block_size));
  }
#endif

  int all_passed = 1;
  for (size_t config_idx = 0; config_idx < config_count; ++config_idx) { all_passed &= check_config(&lhs, &rhs, &reference, &configs[config_idx]); }

  if (all_passed) printf("%s%s all configurations match reference%s\n", COLOR_GREEN, CHECK_MARK, COLOR_RESET);

  tk_mat_free(&lhs);
  tk_mat_free(&rhs);
  tk_mat_free(&reference);
}

static void test_invalid_configs(void) {
  TkMatrix lhs = tk_mat_new(2, 3);
  TkMatrix rhs = tk_mat_new(3, 2);
  TkMatrix out = tk_mat_new(2, 2);

  tk_mat_fill_pattern(&lhs);
  tk_mat_fill_pattern(&rhs);

  tk_matmul_cfg bad_threads = tk_matmul_cfg_new(TK_BACKEND_PTHREAD, TK_LOOP_IKJ, 0, 0, 0, 1);
  CHECK(tk_mat_mul_into(&lhs, &rhs, &out, &bad_threads, NULL) != TK_OK);
  tk_matmul_cfg bad_block = tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IKJ, 1, 0, 1, 0);
  CHECK(tk_mat_mul_into(&lhs, &rhs, &out, &bad_block, NULL) != TK_OK);
  tk_matmul_cfg bad_simd_order = tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IJK, 0, 1, 1, 1);
  CHECK(tk_mat_mul_into(&lhs, &rhs, &out, &bad_simd_order, NULL) != TK_OK);

  if (!tk_sse_available()) {
    tk_matmul_cfg bad_simd = tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IKJ, 0, 1, 1, 1);
    CHECK(tk_mat_mul_into(&lhs, &rhs, &out, &bad_simd, NULL) != TK_OK);
  }

  tk_mat_free(&lhs);
  tk_mat_free(&rhs);
  tk_mat_free(&out);
}

/* Matmul config sweep regression suite, same as the pre-rewrite harness. */
static void test_matmul(void) {
  printf("\n== matmul correctness ==\n");
  test_invalid_configs();

  test_matmul_case(1, 1, 1, 4, 8);
  test_matmul_case(2, 3, 4, 4, 8);
  test_matmul_case(7, 11, 5, 4, 8);
  test_matmul_case(10, 10, 10, 4, 8);
  test_matmul_case(16, 16, 16, 4, 8);
  test_matmul_case(17, 17, 17, 4, 8);
  test_matmul_case(31, 29, 13, 4, 8);

  test_matmul_case(64, 64, 64, 1, 8);
  test_matmul_case(64, 64, 64, 2, 8);
  test_matmul_case(64, 64, 64, 4, 8);
  test_matmul_case(64, 64, 64, 8, 8);

  test_matmul_case(64, 64, 64, 4, 1);
  test_matmul_case(64, 64, 64, 4, 4);
  test_matmul_case(64, 64, 64, 4, 8);
  test_matmul_case(64, 64, 64, 4, 16);
  test_matmul_case(64, 64, 64, 4, 32);

  test_matmul_case(70, 70, 70, 4, 8);
  test_matmul_case(70, 70, 70, 4, 16);
  test_matmul_case(70, 70, 70, 4, 32);

  test_matmul_case(70, 70, 70, 3, 8);
  test_matmul_case(70, 70, 70, 5, 8);
  test_matmul_case(70, 70, 70, 6, 8);

  test_matmul_case(16, 16, 16, 4, 32);
  test_matmul_case(16, 16, 16, 4, 64);
  test_matmul_case(16, 16, 16, 4, 128);

  test_matmul_case(16, 32, 64, 4, 8);
  test_matmul_case(32, 16, 64, 4, 8);
  test_matmul_case(64, 16, 32, 4, 8);
}

/* Tokenizer regression: fixed id list, round-trip, and the out-of-range guard. */
static void test_tokenizer(void) {
  printf("\n== tokenizer regression ==\n");
  const char *input = "Hello, world! This is a Qwen3 test.";
  const uint32_t want[] = {9707, 11, 1879, 0, 1096, 374, 264, 1207, 16948, 18, 1273, 13};
  const size_t want_n = sizeof(want) / sizeof(want[0]);

  TkTokenizer *tz = tk_tokenizer_open(tk_model_default_gguf());
  CHECK(tz != NULL);
  if (!tz) return;

  uint32_t *ids = NULL;
  size_t n = tk_tokenize(tz, input, &ids);
  CHECK(n == want_n);
  if (n == want_n) CHECK(memcmp(ids, want, want_n * sizeof(uint32_t)) == 0);

  char *round = tk_detokenize(tz, ids, n);
  CHECK(strcmp(round, input) == 0);
  free(round);
  if (ids) free(ids);

  /* Out-of-range ids are skipped, not dereferenced. token 0 + 5 decode normally. */
  uint32_t oob[] = {151936, 0, 151936, 5, 151936};
  char *text = tk_detokenize(tz, oob, 5);
  CHECK(text != NULL);
  uint32_t sane[] = {0, 5};
  char *want0 = tk_detokenize(tz, sane, 2);
  CHECK(want0 != NULL && strcmp(text, want0) == 0);
  free(want0);
  free(text);

  tk_tokenizer_close(tz);
}

/* GGUF/model regression: config anchors + tensor presence + dequant spot value. */
static void test_model(void) {
  printf("\n== gguf/model regression ==\n");
  TkGguf *ctx = tk_gguf_open(tk_model_default_gguf());
  CHECK(ctx != NULL);
  if (!ctx) return;

  TkModel cfg = tk_model_from_gguf(ctx);
  CHECK(cfg.vocab_size == 151936);
  CHECK(cfg.num_layers == 28);

  const TkGgufTensor *emb = tk_gguf_tensor_named(ctx, "token_embd.weight");
  CHECK(emb != NULL);

  float buf[1];
  tk_status st = tk_gguf_read(ctx, "token_embd.weight", buf, 0, 1);
  CHECK(st == TK_OK);
  CHECK(buf[0] == -0.00930309296f);

  tk_gguf_close(ctx);
}

int tk_test_all(void) {
  g_failures = 0;
  test_matmul();
  test_tokenizer();
  test_model();

  if (g_failures == 0) {
    printf("\n%s%s all tests passed%s\n", COLOR_GREEN, CHECK_MARK, COLOR_RESET);
  } else {
    printf("\n%s%s %u test failure(s)%s\n", COLOR_RED, CROSS_MARK, g_failures, COLOR_RESET);
  }
  return g_failures == 0 ? 0 : 1;
}
