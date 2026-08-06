#include "tk_backend.h"
#include "tk_bench.h"
#include "tk_common.h"
#include "tk_forward.h"
#include "tk_generate.h"
#include "tk_gguf.h"
#include "tk_model.h"
#include "tk_test.h"
#include "tk_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define QWEN_DEFAULT_GGUF "Qwen3-0.6B-Q8_0.gguf"

static void print_usage(const char *program) {
  fprintf(stderr, "usage: %s [test|bench|all|model|tokenize <text>|detokenize <id...>|forward <text>|generate <text> [options]|bench-infer]\n",
          program);
  fprintf(stderr, "generate options: --seed N --temp T --top-k K --top-p P --n NT --think\n");
  fprintf(stderr, "bench-infer options: --tokens N\n");
}

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

static int run_forward_cmd(int argc, char **argv) {
  const char *text = argc > 2 ? argv[2] : "The capital of France is";
  tk_status s = TK_ERR_INTERNAL;
  TkTokenizer *tz = tk_tokenizer_open(QWEN_DEFAULT_GGUF);
  TkGguf *gguf = NULL;
  uint32_t *ids = NULL;
  TkMatrix logits = {0};
  if (!tz) {
    fprintf(stderr, "error: failed to load tokenizer from %s\n", QWEN_DEFAULT_GGUF);
    return 1;
  }
  gguf = tk_gguf_open(QWEN_DEFAULT_GGUF);
  if (!gguf) {
    tk_tokenizer_close(tz);
    return 1;
  }

  TkModel cfg = tk_model_from_gguf(gguf);
  size_t T = tk_tokenize(tz, text, &ids);
  if (T == 0) {
    fprintf(stderr, "error: empty prompt\n");
    goto out;
  }

  s = tk_forward_logits(gguf, &cfg, ids, T, &logits);
  if (s != TK_OK) {
    fprintf(stderr, "error: forward pass failed (status %d)\n", s);
    goto out;
  }

  /* Dump full logits for the verification script. */
  mkdir("results/data", 0755);
  FILE *f = fopen("results/data/forward_logits.bin", "wb");
  if (f) {
    fwrite(logits.data, sizeof(float), logits.rows * logits.cols, f);
    fclose(f);
  }

  /* Contract output: greedy next-token id per position (predicts token t+1). */
  for (size_t t = 0; t < T; t++) {
    const float *row = logits.data + t * logits.cols;
    size_t best = 0;
    for (size_t i = 1; i < logits.cols; i++)
      if (row[i] > row[best]) best = i;
    printf("%s%zu", t ? " " : "", best);
  }
  printf("\n");

  /* Diagnostics: top-5 final-position candidates + their detokenized text. */
  {
    const float *row = logits.data + (T - 1) * logits.cols;
    uint32_t idx[5] = {0, 0, 0, 0, 0};
    for (size_t i = 0; i < logits.cols; i++)
      for (int b = 0; b < 5; b++)
        if (row[i] > row[idx[b]]) {
          for (int x = 3; x >= b; x--) idx[x + 1] = idx[x];
          idx[b] = i;
          break;
        }
    fprintf(stderr, "top-5 final-position tokens:\n");
    for (int b = 0; b < 5; b++) {
      char *tok = tk_detokenize(tz, &idx[b], 1);
      fprintf(stderr, "  %-7u score=%.4f %s\n", idx[b], row[idx[b]], tok);
      free(tok);
    }
  }

out:
  free(ids);
  tk_mat_free(&logits);
  tk_gguf_close(gguf);
  tk_tokenizer_close(tz);
  return s != TK_OK;
}

static int run_generate_cmd(int argc, char **argv) {
  const char *text = NULL;
  int think = 0;
  uint64_t seed = 42;
  float temp = 0.7f;
  int top_k = 20;
  float top_p = 0.8f;
  size_t max_new = 128;

  for (int i = 2; i < argc; i++) {
    if (argv[i][0] != '-' && !text) {
      text = argv[i];
      continue;
    }
    if (strcmp(argv[i], "--think") == 0) {
      think = 1;
      continue;
    }
    if (i + 1 < argc && strcmp(argv[i], "--seed") == 0) {
      seed = strtoull(argv[++i], NULL, 10);
      continue;
    }
    if (i + 1 < argc && strcmp(argv[i], "--temp") == 0) {
      temp = strtof(argv[++i], NULL);
      continue;
    }
    if (i + 1 < argc && strcmp(argv[i], "--top-k") == 0) {
      top_k = atoi(argv[++i]);
      continue;
    }
    if (i + 1 < argc && strcmp(argv[i], "--top-p") == 0) {
      top_p = strtof(argv[++i], NULL);
      continue;
    }
    if (i + 1 < argc && strcmp(argv[i], "--n") == 0) {
      max_new = strtoull(argv[++i], NULL, 10);
      continue;
    }
    fprintf(stderr, "error: unknown argument %s\n", argv[i]);
    return 1;
  }
  if (!text || max_new == 0) {
    fprintf(stderr, "usage: %s generate <text> [--seed N] [--temp T] [--top-k K] [--top-p P] [--n NT] [--think]\n", argv[0]);
    return 1;
  }

  TkTokenizer *tz = tk_tokenizer_open(QWEN_DEFAULT_GGUF);
  TkGguf *gguf = NULL;
  TkInfer *inf = NULL;
  uint32_t *prompt = NULL, *gen = NULL;
  float *row = NULL;
  tk_status s = TK_OK;
  int rc = 1;
  if (!tz) {
    fprintf(stderr, "error: failed to load tokenizer from %s\n", QWEN_DEFAULT_GGUF);
    return 1;
  }
  if (!(gguf = tk_gguf_open(QWEN_DEFAULT_GGUF))) goto out;
  TkModel cfg = tk_model_from_gguf(gguf);

  size_t pn = 0;
  prompt = tk_chat_prompt(tz, text, think, &pn);
  if (!(inf = tk_infer_new(gguf, &cfg))) {
    fprintf(stderr, "error: failed to initialize inference engine\n");
    goto out;
  }
  row = tk_xmalloc(cfg.vocab_size * sizeof(float));
  gen = tk_xmalloc(max_new * sizeof(uint32_t));
  fprintf(stderr, "prompt: %zu tokens\n", pn);

  for (size_t t = 0; t < pn && s == TK_OK; t++) s = tk_infer_step(inf, prompt[t], row);
  if (s != TK_OK) {
    fprintf(stderr, "error: forward pass failed (status %d)\n", s);
    goto out;
  }

  uint64_t rng = seed;
  size_t gn = 0;
  for (size_t g = 0; g < max_new && s == TK_OK; g++) {
    uint32_t tok = tk_sample(row, cfg.vocab_size, temp, top_k, top_p, &rng);
    /* stop at </|im_end|> (eos) or <|endoftext|> (this config's bos id). */
    if (tok == (uint32_t)cfg.eos_token_id || tok == (uint32_t)cfg.bos_token_id) break;
    gen[gn++] = tok;
    s = tk_infer_step(inf, tok, row);
  }
  if (s != TK_OK) {
    fprintf(stderr, "error: generation failed (status %d)\n", s);
    goto out;
  }

  char *out_text = tk_detokenize(tz, gen, gn);
  printf("%s\n", out_text);
  free(out_text);
  fprintf(stderr, "generated %zu tokens (seed %llu)\n", gn, (unsigned long long)seed);
  rc = 0;

out:
  free(row);
  free(prompt);
  free(gen);
  tk_infer_free(inf);
  tk_gguf_close(gguf);
  tk_tokenizer_close(tz);
  return rc;
}

/* Time real autoregressive tokens (KV cache resident, weights cached once) under
 * different GEMM configs, reporting ms/token and the speedup of the fast config
 * vs the plain scalar baseline. Justifies the Phase 6 kernel choice. */
static int run_bench_infer_cmd(int argc, char **argv) {
  size_t tokens = 30;
  for (int i = 2; i < argc; i++) {
    if (i + 1 < argc && strcmp(argv[i], "--tokens") == 0) {
      tokens = strtoull(argv[++i], NULL, 10);
      continue;
    }
    fprintf(stderr, "usage: %s bench-infer [--tokens N]\n", argv[0]);
    return 1;
  }
  if (tokens == 0) return 1;

  TkTokenizer *tz = tk_tokenizer_open(QWEN_DEFAULT_GGUF);
  TkGguf *gguf = NULL;
  TkInfer *inf = NULL;
  uint32_t *ids = NULL;
  float *row = NULL;
  tk_status s = TK_OK;
  int rc = 1;
  if (!tz) return 1;
  if (!(gguf = tk_gguf_open(QWEN_DEFAULT_GGUF))) goto out;
  TkModel cfg = tk_model_from_gguf(gguf);
  size_t pn = tk_tokenize(tz, "The capital of France is", &ids);
  if (!(inf = tk_infer_new(gguf, &cfg))) goto out;
  row = tk_xmalloc(cfg.vocab_size * sizeof(float));

  /* process the prompt, then warm up a couple of generated tokens */
  for (size_t t = 0; t < pn && s == TK_OK; t++) s = tk_infer_step(inf, ids[t], row);
  for (size_t g = 0; g < 2 && s == TK_OK; g++) s = tk_infer_step(inf, (uint32_t)cfg.bos_token_id, row);
  if (s != TK_OK) goto out;

  tk_matmul_cfg plain = tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IJK, false, false, 1, 64);
  tk_matmul_cfg fast = tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IKJ, false, tk_sse_available(), 1, 64);
  tk_matmul_cfg block = tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IKJ, true, tk_sse_available(), 1, 64);
  const tk_matmul_cfg *cfgs[3] = {&plain, &block, &fast};
  const char *names[3] = {"plain  (IJK scalar)   ", "blocked(IKJ simd)   ", "fast   (IKJ simd)   "};
  double ms[3], tokps[3];

  for (int c = 0; c < 3; c++) {
    tk_infer_set_cfg(inf, cfgs[c]);
    double t0 = tk_now_seconds();
    for (size_t g = 0; g < tokens && s == TK_OK; g++) s = tk_infer_step(inf, (uint32_t)cfg.bos_token_id, row);
    if (s != TK_OK) goto out;
    double dt = tk_now_seconds() - t0;
    ms[c] = dt / (double)tokens * 1000.0;
    tokps[c] = (double)tokens / dt;
    printf("  %s %8.3f ms/token  %7.1f tok/s\n", names[c], ms[c], tokps[c]);
  }
  printf("  fast speedup vs plain: %.2fx, vs blocked: %.2fx\n", ms[0] / ms[2], ms[1] / ms[2]);
  rc = 0;

out:
  free(row);
  free(ids);
  tk_infer_free(inf);
  tk_gguf_close(gguf);
  tk_tokenizer_close(tz);
  return rc;
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
  if (strcmp(mode, "forward") == 0) return run_forward_cmd(argc, argv);
  if (strcmp(mode, "generate") == 0) return run_generate_cmd(argc, argv);
  if (strcmp(mode, "bench-infer") == 0) return run_bench_infer_cmd(argc, argv);

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
