#include "tk_forward.h"
#include "tk_backend.h"
#include "tk_common.h"
#include "tk_gguf.h"
#include "tk_matrix.h"
#include "tk_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- weight loading ---- */

/* Load tensor `name` (GGUF row-major [out x in], i.e. an nn.Linear weight) into
 * Wt as a transposed row-major (in x out) matrix so that y = x @ Wt. Q8_0/F32
 * are dequantized by the gguf layer. */
static tk_status load_weight(const TkGguf *g, const char *name, size_t in, size_t out, TkMatrix *Wt) {
  *Wt = tk_mat_new(in, out);
  float *tmp = tk_xmalloc(out * in * sizeof(float));
  tk_status s = tk_gguf_read(g, name, tmp, 0, out * in);
  if (s == TK_OK)
    for (size_t o = 0; o < out; o++)
      for (size_t k = 0; k < in; k++) Wt->data[k * out + o] = tmp[o * in + k];
  free(tmp);
  if (s != TK_OK) tk_mat_free(Wt);
  return s;
}

/* ---- normalization ---- */

static void rms_norm(const float *x, const float *w, size_t n, float eps, float *y) {
  float ss = 0.0f;
  for (size_t i = 0; i < n; i++) ss += x[i] * x[i];
  float r = 1.0f / sqrtf(ss / (float)n + eps);
  for (size_t i = 0; i < n; i++) y[i] = x[i] * r * w[i];
}

/* Qwen3 q_norm/k_norm: RMSNorm over each contiguous head_dim block of a
 * (rows, heads, hd) tensor held as (rows, heads*hd) row-major. */
static void head_rms_norm(float *x, const float *w, size_t rows, size_t heads, size_t hd, float eps) {
  for (size_t r = 0; r < rows; r++)
    for (size_t h = 0; h < heads; h++) rms_norm(x + (r * heads + h) * hd, w, hd, eps, x + (r * heads + h) * hd);
}

/* ---- RoPE ---- */

/* Precompute the half-split rotary cos/sin tables for positions 0..T-1. */
static void rope_table(float *cos_t, float *sin_t, size_t T, size_t hd, float theta) {
  size_t half = hd / 2;
  for (size_t r = 0; r < T; r++)
    for (size_t i = 0; i < half; i++) {
      float ang = (float)r * powf(theta, -(float)(2 * i) / (float)hd);
      cos_t[r * half + i] = cosf(ang);
      sin_t[r * half + i] = sinf(ang);
    }
}

/* Apply rotary (HF/Qwen half-split convention) to each head block in place. */
static void apply_rope(float *qk, size_t rows, size_t heads, size_t hd, const float *cos_t, const float *sin_t) {
  size_t half = hd / 2;
  for (size_t r = 0; r < rows; r++) {
    const float *cr = cos_t + r * half, *sr = sin_t + r * half;
    for (size_t h = 0; h < heads; h++) {
      float *x = qk + (r * heads + h) * hd;
      for (size_t i = 0; i < half; i++) {
        float a = x[i], b = x[i + half];
        x[i] = cr[i] * a - sr[i] * b;
        x[i + half] = sr[i] * a + cr[i] * b;
      }
    }
  }
}

/* ---- GQA attention (explicit loops; causal) ---- */
static void attention_heads(float *out, const float *q, const float *k, const float *v, size_t T, size_t n_heads, size_t n_kv, size_t hd) {
  size_t qstride = n_heads * hd, kstride = n_kv * hd;
  float rscale = 1.0f / sqrtf((float)hd);
  float *scores = tk_xmalloc(T * T * sizeof(float));
  float *p = tk_xmalloc(T * sizeof(float));

  for (size_t h = 0; h < n_heads; h++) {
    size_t kvh = h / (n_heads / n_kv);
    const float *kh = k + kvh * hd;
    const float *vh = v + kvh * hd;
    float *oh = out + h * hd;

    for (size_t m = 0; m < T; m++) {
      const float *qm = q + m * qstride + h * hd;
      float *sm = scores + m * T;
      for (size_t n = 0; n < T; n++) {
        if (n > m) {
          sm[n] = -INFINITY;
          continue;
        }
        const float *kn = kh + n * kstride;
        float s = 0.0f;
        for (size_t d = 0; d < hd; d++) s += qm[d] * kn[d];
        sm[n] = s * rscale;
      }
      float mx = sm[0];
      for (size_t n = 1; n < T; n++)
        if (sm[n] > mx) mx = sm[n];
      float sum = 0.0f;
      for (size_t n = 0; n < T; n++) {
        p[n] = expf(sm[n] - mx);
        sum += p[n];
      }
      for (size_t n = 0; n < T; n++) p[n] /= sum;

      float *om = oh + m * qstride;
      for (size_t d = 0; d < hd; d++) {
        float acc = 0.0f;
        for (size_t n = 0; n < T; n++) acc += p[n] * vh[n * kstride + d];
        om[d] = acc;
      }
    }
  }
  free(scores);
  free(p);
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

tk_status tk_forward_logits(const TkGguf *g, const TkModel *c, const uint32_t *ids, size_t T, TkMatrix *logits) {
  const size_t hidden = c->hidden_size, vocab = c->vocab_size, hd = c->head_dim;
  const size_t n_heads = c->num_attention_heads, n_kv = c->num_kv_heads;
  const size_t qwidth = n_heads * hd, kvwidth = n_kv * hd, inter = c->intermediate_size;
  const float eps = c->rms_norm_eps;

  /* IPJK/blocking-free single backend: scalar dot-product accumulation that most
   * closely matches the reference, so logits line up to ~float rounding. */
  tk_matmul_cfg cfg = tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IJK, false, false, 1, 64);
  tk_status status;

  /* Tied embedding (hidden x vocab); also serves as the lm_head projection. */
  TkMatrix emb;
  if ((status = load_weight(g, "token_embd.weight", hidden, vocab, &emb)) != TK_OK) return status;

  TkMatrix H = tk_mat_new(T, hidden);
  for (size_t t = 0; t < T; t++)
    for (size_t k = 0; k < hidden; k++) H.data[t * hidden + k] = emb.data[k * vocab + ids[t]];

  float *cos_t = tk_xmalloc(T * (hd / 2) * sizeof(float));
  float *sin_t = tk_xmalloc(T * (hd / 2) * sizeof(float));
  rope_table(cos_t, sin_t, T, hd, c->rope_theta);

  char name[64];
  TkMatrix q = tk_mat_new(T, qwidth);
  TkMatrix k = tk_mat_new(T, kvwidth);
  TkMatrix v = tk_mat_new(T, kvwidth);
  TkMatrix o = tk_mat_new(T, qwidth);
  TkMatrix ao = tk_mat_new(T, hidden);
  TkMatrix xn = tk_mat_new(T, hidden);
  TkMatrix xf = tk_mat_new(T, hidden);
  TkMatrix gate = tk_mat_new(T, inter);
  TkMatrix up = tk_mat_new(T, inter);
  TkMatrix act = tk_mat_new(T, inter);
  TkMatrix down = tk_mat_new(T, hidden);
  float *norm_w = tk_xmalloc(hidden * sizeof(float));
  float *head_w = tk_xmalloc(hd * sizeof(float));

  for (size_t l = 0; l < c->num_layers; l++) {
    /* attn norm */
    snprintf(name, sizeof(name), "blk.%zu.attn_norm.weight", l);
    if ((status = tk_gguf_read(g, name, norm_w, 0, hidden)) != TK_OK) goto fail;
    for (size_t t = 0; t < T; t++) rms_norm(H.data + t * hidden, norm_w, hidden, eps, xn.data + t * hidden);

    /* QKV projections */
    TkMatrix Wq, Wk, Wv;
    snprintf(name, sizeof(name), "blk.%zu.attn_q.weight", l);
    if ((status = load_weight(g, name, hidden, qwidth, &Wq)) != TK_OK) goto fail;
    snprintf(name, sizeof(name), "blk.%zu.attn_k.weight", l);
    if ((status = load_weight(g, name, hidden, kvwidth, &Wk)) != TK_OK) goto fail;
    snprintf(name, sizeof(name), "blk.%zu.attn_v.weight", l);
    if ((status = load_weight(g, name, hidden, kvwidth, &Wv)) != TK_OK) goto fail;
    (void)tk_mat_mul_into(&xn, &Wq, &q, &cfg, &status);
    (void)tk_mat_mul_into(&xn, &Wk, &k, &cfg, &status);
    (void)tk_mat_mul_into(&xn, &Wv, &v, &cfg, &status);
    tk_mat_free(&Wq);
    tk_mat_free(&Wk);
    tk_mat_free(&Wv);

    /* per-head QK norm, then RoPE */
    snprintf(name, sizeof(name), "blk.%zu.attn_q_norm.weight", l);
    if ((status = tk_gguf_read(g, name, head_w, 0, hd)) != TK_OK) goto fail;
    head_rms_norm(q.data, head_w, T, n_heads, hd, eps);
    snprintf(name, sizeof(name), "blk.%zu.attn_k_norm.weight", l);
    if ((status = tk_gguf_read(g, name, head_w, 0, hd)) != TK_OK) goto fail;
    head_rms_norm(k.data, head_w, T, n_kv, hd, eps);
    apply_rope(q.data, T, n_heads, hd, cos_t, sin_t);
    apply_rope(k.data, T, n_kv, hd, cos_t, sin_t);

    /* GQA attention -> output projection -> residual */
    attention_heads(o.data, q.data, k.data, v.data, T, n_heads, n_kv, hd);
    TkMatrix Wo;
    snprintf(name, sizeof(name), "blk.%zu.attn_output.weight", l);
    if ((status = load_weight(g, name, qwidth, hidden, &Wo)) != TK_OK) goto fail;
    (void)tk_mat_mul_into(&o, &Wo, &ao, &cfg, &status);
    tk_mat_free(&Wo);
    for (size_t i = 0; i < T * hidden; i++) H.data[i] += ao.data[i];

    /* SwiGLU MLP -> residual */
    snprintf(name, sizeof(name), "blk.%zu.ffn_norm.weight", l);
    if ((status = tk_gguf_read(g, name, norm_w, 0, hidden)) != TK_OK) goto fail;
    for (size_t t = 0; t < T; t++) rms_norm(H.data + t * hidden, norm_w, hidden, eps, xf.data + t * hidden);
    TkMatrix Wg, Wu, Wd;
    snprintf(name, sizeof(name), "blk.%zu.ffn_gate.weight", l);
    if ((status = load_weight(g, name, hidden, inter, &Wg)) != TK_OK) goto fail;
    snprintf(name, sizeof(name), "blk.%zu.ffn_up.weight", l);
    if ((status = load_weight(g, name, hidden, inter, &Wu)) != TK_OK) goto fail;
    snprintf(name, sizeof(name), "blk.%zu.ffn_down.weight", l);
    if ((status = load_weight(g, name, inter, hidden, &Wd)) != TK_OK) goto fail;
    (void)tk_mat_mul_into(&xf, &Wg, &gate, &cfg, &status);
    (void)tk_mat_mul_into(&xf, &Wu, &up, &cfg, &status);
    for (size_t i = 0; i < T * inter; i++) act.data[i] = silu(gate.data[i]) * up.data[i];
    (void)tk_mat_mul_into(&act, &Wd, &down, &cfg, &status);
    tk_mat_free(&Wg);
    tk_mat_free(&Wu);
    tk_mat_free(&Wd);
    for (size_t i = 0; i < T * hidden; i++) H.data[i] += down.data[i];
  }

  /* final norm + lm_head = tied embedding */
  if ((status = tk_gguf_read(g, "output_norm.weight", norm_w, 0, hidden)) != TK_OK) goto fail;
  for (size_t t = 0; t < T; t++) rms_norm(H.data + t * hidden, norm_w, hidden, eps, xf.data + t * hidden);
  *logits = tk_mat_new(T, vocab);
  (void)tk_mat_mul_into(&xf, &emb, logits, &cfg, &status);

  status = TK_OK;
fail:
  tk_mat_free(&emb);
  tk_mat_free(&H);
  free(cos_t);
  free(sin_t);
  tk_mat_free(&q);
  tk_mat_free(&k);
  tk_mat_free(&v);
  tk_mat_free(&o);
  tk_mat_free(&ao);
  tk_mat_free(&xn);
  tk_mat_free(&xf);
  tk_mat_free(&gate);
  tk_mat_free(&up);
  tk_mat_free(&act);
  tk_mat_free(&down);
  free(norm_w);
  free(head_w);
  return status;
}
