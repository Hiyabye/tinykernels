#include "tk_generate.h"
#include "tk_common.h"
#include "tk_tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- sampling ---- */

static uint64_t xorshift64(uint64_t *s) {
  uint64_t x = *s;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  return *s = x;
}

typedef struct {
  float v;
  uint32_t idx;
} Cand;

static int cand_cmp_desc(const void *a, const void *b) {
  float va = ((const Cand *)a)->v, vb = ((const Cand *)b)->v;
  return (va > vb) ? -1 : (va < vb) ? 1 : 0;
}

uint32_t tk_sample(const float *logits, size_t vocab, float temperature, int top_k, float top_p, uint64_t *rng) {
  if (temperature <= 0.0f) { /* greedy */
    uint32_t best = 0;
    float bv = -INFINITY;
    for (size_t i = 0; i < vocab; i++)
      if (logits[i] > bv) {
        bv = logits[i];
        best = (uint32_t)i;
      }
    return best;
  }

  Cand *c = tk_xmalloc(vocab * sizeof(Cand));
  for (size_t i = 0; i < vocab; i++) {
    c[i].v = logits[i] / temperature;
    c[i].idx = (uint32_t)i;
  }
  qsort(c, vocab, sizeof(Cand), cand_cmp_desc);

  size_t j = (top_k > 0 && (size_t)top_k < vocab) ? (size_t)top_k : vocab;
  float mx = c[0].v;
  double sum = 0.0, kept_sum = 0.0;
  for (size_t i = 0; i < j; i++) {
    c[i].v = expf(c[i].v - mx);
    kept_sum += c[i].v;
  }
  sum = kept_sum;

  size_t m = j;
  if (top_p > 0.0f && top_p < 1.0f) { /* nucleus: keep the smallest prefix whose mass >= top_p */
    double cum = 0.0;
    for (m = 0; m < j; m++) {
      cum += c[m].v;
      if (cum >= top_p * kept_sum) break;
    }
    m++; /* include the token that crossed the threshold */
    kept_sum = 0.0;
    for (size_t i = 0; i < m; i++) kept_sum += c[i].v;
    sum = kept_sum;
  }

  double r = (double)(xorshift64(rng) >> 11) / 9007199254740992.0 * sum;
  double acc = 0.0;
  uint32_t out = c[0].idx;
  for (size_t i = 0; i < m; i++) {
    acc += c[i].v;
    if (acc >= r) {
      out = c[i].idx;
      break;
    }
  }
  free(c);
  return out;
}

/* ---- chat template ---- */

uint32_t *tk_chat_prompt(const struct TkTokenizer *tz, const char *user, int think, size_t *n) {
  static const char *sys = "You are Qwen, a helpful AI assistant.";
  size_t cap = strlen(sys) + strlen(user) + 256;
  char *buf = tk_xmalloc(cap);
  snprintf(buf, cap,
           "<|im_start|>system\n%s<|im_end|>\n"
           "<|im_start|>user\n%s<|im_end|>\n"
           "<|im_start|>assistant\n",
           sys, user);

  uint32_t *ids;
  size_t cnt = tk_tokenize(tz, buf, &ids);
  if (think) {
    /* The ` thinking` token is an added (non-mergeable) special token that BPE
     * collapses to the normal word token; append its id verbatim. */
    uint32_t tid = tk_token_id(tz, " thinking");
    if (tid != UINT32_MAX) {
      ids = tk_xrealloc(ids, (cnt + 1) * sizeof(uint32_t));
      ids[cnt++] = tid;
    }
  }
  free(buf);
  *n = cnt;
  return ids;
}
