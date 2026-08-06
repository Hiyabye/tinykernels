#include "tk_bench.h"
#include "tk_gguf.h"
#include "tk_model.h"
#include "tk_test.h"
#include "tk_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QWEN_DEFAULT_GGUF "Qwen3-0.6B-Q8_0.gguf"

static void print_usage(const char *program) { fprintf(stderr, "usage: %s [test|bench|all|model|tokenize <text>|detokenize <id...>]\n", program); }

static int run_tokenizer_cmd(int argc, char **argv) {
  TkTokenizer *tz = tk_tokenizer_open(QWEN_DEFAULT_GGUF);
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
    size_t n = tk_tokenize(tz, argv[2], &ids);
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
    char *text = tk_detokenize(tz, ids, n);
    printf("%s\n", text);
    free(text);
    free(ids);
  }

out:
  tk_tokenizer_close(tz);
  return rc;
}

static int check_tensor(const TkGguf *ctx, const char *name, size_t want_nelems) {
  const TkGgufTensor *t = tk_gguf_tensor_named(ctx, name);
  if (!t) {
    fprintf(stderr, "  FAIL missing tensor %s\n", name);
    return 0;
  }
  if (t->n_elems != want_nelems) {
    fprintf(stderr, "  FAIL shape %s: %zu elements, expected %zu\n", name, t->n_elems, want_nelems);
    return 0;
  }
  return 1;
}

static int run_model_cmd(const char *path) {
  TkGguf *ctx = tk_gguf_open(path);
  if (!ctx) {
    fprintf(stderr, "error: failed to load GGUF from %s\n", path);
    return 1;
  }

  TkModel cfg = tk_model_from_gguf(ctx);
  tk_model_print(&cfg);

  /* Validate the full tensor inventory against the derived config. */
  printf("\ntensor inventory: %zu tensors in file\n", tk_gguf_tensor_count(ctx));
  int ok = 1;
  ok &= check_tensor(ctx, "token_embd.weight", cfg.vocab_size * cfg.hidden_size);
  ok &= check_tensor(ctx, "output_norm.weight", cfg.hidden_size);
  size_t head_total = cfg.num_attention_heads * cfg.head_dim;
  size_t kv_total = cfg.num_kv_heads * cfg.head_dim;
  for (size_t l = 0; l < cfg.num_layers; l++) {
    char name[64];
    size_t inter = cfg.intermediate_size;
    snprintf(name, sizeof(name), "blk.%zu.attn_norm.weight", l);
    ok &= check_tensor(ctx, name, cfg.hidden_size);
    snprintf(name, sizeof(name), "blk.%zu.attn_q.weight", l);
    ok &= check_tensor(ctx, name, cfg.hidden_size * head_total);
    snprintf(name, sizeof(name), "blk.%zu.attn_k.weight", l);
    ok &= check_tensor(ctx, name, cfg.hidden_size * kv_total);
    snprintf(name, sizeof(name), "blk.%zu.attn_v.weight", l);
    ok &= check_tensor(ctx, name, cfg.hidden_size * kv_total);
    snprintf(name, sizeof(name), "blk.%zu.attn_output.weight", l);
    ok &= check_tensor(ctx, name, head_total * cfg.hidden_size);
    snprintf(name, sizeof(name), "blk.%zu.attn_q_norm.weight", l);
    ok &= check_tensor(ctx, name, cfg.head_dim);
    snprintf(name, sizeof(name), "blk.%zu.attn_k_norm.weight", l);
    ok &= check_tensor(ctx, name, cfg.head_dim);
    snprintf(name, sizeof(name), "blk.%zu.ffn_norm.weight", l);
    ok &= check_tensor(ctx, name, cfg.hidden_size);
    snprintf(name, sizeof(name), "blk.%zu.ffn_gate.weight", l);
    ok &= check_tensor(ctx, name, cfg.hidden_size * inter);
    snprintf(name, sizeof(name), "blk.%zu.ffn_up.weight", l);
    ok &= check_tensor(ctx, name, cfg.hidden_size * inter);
    snprintf(name, sizeof(name), "blk.%zu.ffn_down.weight", l);
    ok &= check_tensor(ctx, name, inter * cfg.hidden_size);
  }

  /* Spot-check dequantized weight values (F32 embedding + one Q8_0 block). */
  float v0, vq;
  tk_status d0 = tk_gguf_read(ctx, "token_embd.weight", &v0, 0, 1);
  tk_status dq = tk_gguf_read(ctx, "blk.0.attn_q.weight", &vq, 0, 1);
  if (d0 == TK_OK && dq == TK_OK && ok) {
    printf("\nvalidated OK\n");
    printf("spot-check token_embd.weight[0] = %.9g\n", v0);
    printf("spot-check blk.0.attn_q.weight[0] (Q8_0) = %.9g\n", vq);
  } else {
    fprintf(stderr, "\nvalidation FAILED (dequant d0=%d dq=%d)\n", d0, dq);
  }

  tk_gguf_close(ctx);
  return (d0 == TK_OK && dq == TK_OK && ok) ? 0 : 1;
}

static int run_tests(void) {
  if (tk_test_all() != 0) {
    fprintf(stderr, "tests failed\n");
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
    tk_bench_default_suite("results/data/benchmark_results.csv");
    return 0;
  }

  if (strcmp(mode, "all") == 0) {
    if (run_tests() != 0) return 1;
    tk_bench_default_suite("results/data/benchmark_results.csv");
    return 0;
  }

  if (strcmp(mode, "model") == 0) return run_model_cmd(QWEN_DEFAULT_GGUF);

  print_usage(argv[0]);
  return 1;
}
