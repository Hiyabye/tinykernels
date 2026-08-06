#include "bench_matmul.h"
#include "qwen.h"
#include "qwen_tokenizer.h"
#include "test_matmul.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QWEN_DEFAULT_GGUF "Qwen3-0.6B-Q8_0.gguf"

static void print_usage(const char *program) { fprintf(stderr, "usage: %s [test|bench|all|model|tokenize <text>|detokenize <id...>]\n", program); }

static int run_tokenizer_cmd(int argc, char **argv) {
  QwenTokenizer *tz = qwen_tokenizer_load(QWEN_DEFAULT_GGUF);
  if (!tz) {
    fprintf(stderr, "error: failed to load tokenizer from %s\n", QWEN_DEFAULT_GGUF);
    return 1;
  }

  int rc = 0;
  if (strcmp(argv[1], "tokenize") == 0) {
    if (argc < 3) {
      fprintf(stderr, "usage: %s tokenize <text>\n", argv[0]);
      rc = 1;
      goto out;
    }
    uint32_t *ids = NULL;
    size_t n = qwen_tokenize(tz, argv[2], &ids);
    for (size_t i = 0; i < n; i++) printf("%s%u", i ? " " : "", ids[i]);
    printf("\n");
    free(ids);
  } else {
    if (argc < 3) {
      fprintf(stderr, "usage: %s detokenize <id...>\n", argv[0]);
      rc = 1;
      goto out;
    }
    size_t n = (size_t)(argc - 2);
    uint32_t *ids = malloc(n * sizeof(uint32_t));
    for (size_t i = 0; i < n; i++) ids[i] = (uint32_t)strtoul(argv[i + 2], NULL, 10);
    char *text = qwen_detokenize(tz, ids, n);
    printf("%s\n", text);
    free(text);
    free(ids);
  }

out:
  qwen_tokenizer_free(tz);
  return rc;
}

static int run_tests(void) {
  if (!test_matmul_correctness()) {
    fprintf(stderr, "correctness tests failed\n");
    return 1;
  }

  return 0;
}

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "test";

  if (strcmp(mode, "tokenize") == 0 || strcmp(mode, "detokenize") == 0) { return run_tokenizer_cmd(argc, argv); }

  if (argc > 2) {
    print_usage(argv[0]);
    return 1;
  }

  if (strcmp(mode, "test") == 0) return run_tests();

  if (strcmp(mode, "bench") == 0) {
    bench_run_default_suite("results/data/benchmark_results.csv");
    return 0;
  }

  if (strcmp(mode, "all") == 0) {
    if (run_tests() != 0) return 1;
    bench_run_default_suite("results/data/benchmark_results.csv");
    return 0;
  }

  if (strcmp(mode, "model") == 0) {
    QwenConfig cfg = qwen_config_qwen3_0_6b();
    qwen_config_print(&cfg);
    return 0;
  }

  print_usage(argv[0]);
  return 1;
}
