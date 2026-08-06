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

/* Apply rotary (HF/Qwen half-split convention) in place to `rows` heads at the
 * single position whose precomputed (half) cos/sin tables are cr/sr. */
static void apply_rope(float *qk, size_t rows, size_t heads, size_t hd, const float *cr, const float *sr) {
  size_t half = hd / 2;
  for (size_t r = 0; r < rows; r++) {
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

/* ---- GQA attention for the newest (position `n`) query. `k`/`v` hold the
 * cached (n+1, kvwidth) tensors; `q` is the single (qwidth) query row. Output
 * for each head goes to `out` (qwidth). `scratch` holds n+1 floats. */
static void attention_step(float *out, const float *q, const float *k, const float *v, size_t n, size_t n_heads, size_t n_kv, size_t hd,
                           float *scratch) {
  size_t kstride = n_kv * hd;
  float rscale = 1.0f / sqrtf((float)hd);

  for (size_t h = 0; h < n_heads; h++) {
    size_t kvh = h / (n_heads / n_kv);
    const float *kh = k + kvh * hd, *vh = v + kvh * hd;
    const float *qm = q + h * hd;
    float mx = -INFINITY;
    for (size_t nn = 0; nn <= n; nn++) {
      const float *kn = kh + nn * kstride;
      float s = 0.0f;
      for (size_t d = 0; d < hd; d++) s += qm[d] * kn[d];
      scratch[nn] = s * rscale;
      if (scratch[nn] > mx) mx = scratch[nn];
    }
    float sum = 0.0f;
    for (size_t nn = 0; nn <= n; nn++) {
      scratch[nn] = expf(scratch[nn] - mx);
      sum += scratch[nn];
    }
    float *om = out + h * hd;
    for (size_t d = 0; d < hd; d++) {
      float acc = 0.0f;
      for (size_t nn = 0; nn <= n; nn++) acc += scratch[nn] * vh[nn * kstride + d];
      om[d] = acc / sum;
    }
  }
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

/* ---- Inference engine (KV cache) ---- */

#define KV_GROW_MIN 64
#define INFER_BLOCK 64

/* Resident per-layer dequantized weights. Each `w*` is row-major [in x out]
 * (transposed from GGUF) so y = x @ W; norms are plain vectors. Cached once at
 * tk_infer_new so the per-token step does zero file I/O / dequant. */
typedef struct {
  TkMatrix wq, wk, wv, wo;     /* attn: [hidden x q/kvwidth], [q/kvwidth x hidden] */
  TkMatrix wgate, wup, wdown;  /* ffn: [hidden x inter], [inter x hidden] */
  float *attn_norm, *ffn_norm; /* hidden */
  float *q_norm, *k_norm;      /* hd */
} LayerWt;

struct TkInfer {
  const TkGguf *g;
  const TkModel *c;
  size_t n;   /* tokens processed = filled KV rows */
  size_t cap; /* KV cache allocated rows per layer */
  tk_matmul_cfg cfg;

  TkMatrix emb;         /* hidden x vocab, tied lm_head, resident */
  TkMatrix *kc, *vc;    /* per-layer K/V caches, rows=cap, cols=kvwidth */
  float *cos_t, *sin_t; /* per-position rope tables (cap * hd/2) */
  float *prow;          /* attention softmax scratch (cap floats) */

  /* per-step scratch (1 x width matrices) */
  TkMatrix H, xn, q, k, v, o, ao, xf, gate, up, act, down;

  /* resident dequantized weights (loaded once at init) */
  LayerWt *layers;   /* num_layers */
  float *out_norm_w; /* hidden, final RMSNorm */
};

static tk_status kv_grow(TkInfer *inf, size_t need) {
  if (need <= inf->cap) return TK_OK;
  size_t new_cap = inf->cap ? inf->cap * 2 : KV_GROW_MIN;
  while (new_cap < need) new_cap *= 2;
  size_t kvwidth = inf->c->num_kv_heads * inf->c->head_dim;
  size_t half = inf->c->head_dim / 2;
  for (size_t l = 0; l < inf->c->num_layers; l++) {
    inf->kc[l].data = tk_xrealloc(inf->kc[l].data, new_cap * kvwidth * sizeof(float));
    inf->vc[l].data = tk_xrealloc(inf->vc[l].data, new_cap * kvwidth * sizeof(float));
  }
  inf->cos_t = tk_xrealloc(inf->cos_t, new_cap * half * sizeof(float));
  inf->sin_t = tk_xrealloc(inf->sin_t, new_cap * half * sizeof(float));
  inf->prow = tk_xrealloc(inf->prow, new_cap * sizeof(float));
  inf->cap = new_cap;
  return TK_OK;
}

TkInfer *tk_infer_new(const TkGguf *g, const TkModel *c) {
  TkInfer *inf = tk_xcalloc(1, sizeof(TkInfer));
  inf->g = g;
  inf->c = c;

  size_t L = c->num_layers, kvwidth = c->num_kv_heads * c->head_dim;
  inf->kc = tk_xcalloc(L, sizeof(TkMatrix));
  inf->vc = tk_xcalloc(L, sizeof(TkMatrix));
  for (size_t l = 0; l < L; l++) {
    inf->kc[l] = (TkMatrix){0, kvwidth, NULL};
    inf->vc[l] = (TkMatrix){0, kvwidth, NULL};
  }

  if (load_weight(g, "token_embd.weight", c->hidden_size, c->vocab_size, &inf->emb) != TK_OK) {
    tk_infer_free(inf);
    return NULL;
  }

  /* Cache every weight resident so the per-token step does no file I/O. */
  inf->cfg = tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IKJ, false, tk_sse_available(), 1, INFER_BLOCK);
  size_t hidden = c->hidden_size, qwidth = c->num_attention_heads * c->head_dim;
  size_t inter = c->intermediate_size, hd = c->head_dim;
  inf->layers = tk_xcalloc(c->num_layers, sizeof(LayerWt));
  char name[64];
  for (size_t l = 0; l < c->num_layers; l++) {
    LayerWt *lt = &inf->layers[l];
    snprintf(name, sizeof(name), "blk.%zu.attn_q.weight", l);
    if (load_weight(g, name, hidden, qwidth, &lt->wq) != TK_OK) goto fail;
    snprintf(name, sizeof(name), "blk.%zu.attn_k.weight", l);
    if (load_weight(g, name, hidden, kvwidth, &lt->wk) != TK_OK) goto fail;
    snprintf(name, sizeof(name), "blk.%zu.attn_v.weight", l);
    if (load_weight(g, name, hidden, kvwidth, &lt->wv) != TK_OK) goto fail;
    snprintf(name, sizeof(name), "blk.%zu.attn_output.weight", l);
    if (load_weight(g, name, qwidth, hidden, &lt->wo) != TK_OK) goto fail;
    snprintf(name, sizeof(name), "blk.%zu.ffn_gate.weight", l);
    if (load_weight(g, name, hidden, inter, &lt->wgate) != TK_OK) goto fail;
    snprintf(name, sizeof(name), "blk.%zu.ffn_up.weight", l);
    if (load_weight(g, name, hidden, inter, &lt->wup) != TK_OK) goto fail;
    snprintf(name, sizeof(name), "blk.%zu.ffn_down.weight", l);
    if (load_weight(g, name, inter, hidden, &lt->wdown) != TK_OK) goto fail;
    lt->attn_norm = tk_xmalloc(hidden * sizeof(float));
    lt->ffn_norm = tk_xmalloc(hidden * sizeof(float));
    lt->q_norm = tk_xmalloc(hd * sizeof(float));
    lt->k_norm = tk_xmalloc(hd * sizeof(float));
    snprintf(name, sizeof(name), "blk.%zu.attn_norm.weight", l);
    if (tk_gguf_read(g, name, lt->attn_norm, 0, hidden) != TK_OK) goto fail;
    snprintf(name, sizeof(name), "blk.%zu.ffn_norm.weight", l);
    if (tk_gguf_read(g, name, lt->ffn_norm, 0, hidden) != TK_OK) goto fail;
    snprintf(name, sizeof(name), "blk.%zu.attn_q_norm.weight", l);
    if (tk_gguf_read(g, name, lt->q_norm, 0, hd) != TK_OK) goto fail;
    snprintf(name, sizeof(name), "blk.%zu.attn_k_norm.weight", l);
    if (tk_gguf_read(g, name, lt->k_norm, 0, hd) != TK_OK) goto fail;
  }
  inf->out_norm_w = tk_xmalloc(hidden * sizeof(float));
  if (tk_gguf_read(g, "output_norm.weight", inf->out_norm_w, 0, hidden) != TK_OK) goto fail;

  inf->H = tk_mat_new(1, hidden);
  inf->xn = tk_mat_new(1, hidden);
  inf->q = tk_mat_new(1, qwidth);
  inf->k = tk_mat_new(1, kvwidth);
  inf->v = tk_mat_new(1, kvwidth);
  inf->o = tk_mat_new(1, qwidth);
  inf->ao = tk_mat_new(1, hidden);
  inf->xf = tk_mat_new(1, hidden);
  inf->gate = tk_mat_new(1, inter);
  inf->up = tk_mat_new(1, inter);
  inf->act = tk_mat_new(1, inter);
  inf->down = tk_mat_new(1, hidden);
  return inf;

fail:
  tk_infer_free(inf);
  return NULL;
}

void tk_infer_free(TkInfer *inf) {
  if (!inf) return;
  for (size_t l = 0; l < inf->c->num_layers; l++) {
    free(inf->kc[l].data);
    free(inf->vc[l].data);
    if (inf->layers) {
      LayerWt *lt = &inf->layers[l];
      tk_mat_free(&lt->wq);
      tk_mat_free(&lt->wk);
      tk_mat_free(&lt->wv);
      tk_mat_free(&lt->wo);
      tk_mat_free(&lt->wgate);
      tk_mat_free(&lt->wup);
      tk_mat_free(&lt->wdown);
      free(lt->attn_norm);
      free(lt->ffn_norm);
      free(lt->q_norm);
      free(lt->k_norm);
    }
  }
  free(inf->kc);
  free(inf->vc);
  free(inf->layers);
  free(inf->out_norm_w);
  tk_mat_free(&inf->emb);
  tk_mat_free(&inf->H);
  tk_mat_free(&inf->xn);
  tk_mat_free(&inf->q);
  tk_mat_free(&inf->k);
  tk_mat_free(&inf->v);
  tk_mat_free(&inf->o);
  tk_mat_free(&inf->ao);
  tk_mat_free(&inf->xf);
  tk_mat_free(&inf->gate);
  tk_mat_free(&inf->up);
  tk_mat_free(&inf->act);
  tk_mat_free(&inf->down);
  free(inf->cos_t);
  free(inf->sin_t);
  free(inf->prow);
  free(inf);
}

size_t tk_infer_len(const TkInfer *inf) { return inf->n; }

void tk_infer_set_cfg(TkInfer *inf, const tk_matmul_cfg *cfg) {
  if (inf && cfg) inf->cfg = *cfg;
}

tk_status tk_infer_step(TkInfer *inf, uint32_t id, float *logits) {
  const TkModel *c = inf->c;
  const size_t hidden = c->hidden_size, vocab = c->vocab_size, hd = c->head_dim;
  const size_t n_heads = c->num_attention_heads, n_kv = c->num_kv_heads;
  const size_t kvwidth = n_kv * hd, inter = c->intermediate_size;
  const size_t half = hd / 2;
  const float eps = c->rms_norm_eps;
  const size_t n = inf->n;
  tk_status status;

  tk_matmul_cfg cfg = inf->cfg;
  if ((status = kv_grow(inf, n + 1)) != TK_OK) return status;

  /* embedding lookup for this position */
  for (size_t k = 0; k < hidden; k++) inf->H.data[k] = inf->emb.data[k * vocab + (size_t)id];

  /* rope tables for this position */
  for (size_t i = 0; i < half; i++) {
    float ang = (float)n * powf(c->rope_theta, -(float)(2 * i) / (float)hd);
    inf->cos_t[n * half + i] = cosf(ang);
    inf->sin_t[n * half + i] = sinf(ang);
  }

  for (size_t l = 0; l < c->num_layers; l++) {
    const LayerWt *lt = &inf->layers[l];

    /* attention norm */
    rms_norm(inf->H.data, lt->attn_norm, hidden, eps, inf->xn.data);

    /* QKV projections (resident weights) */
    (void)tk_mat_mul_into(&inf->xn, &lt->wq, &inf->q, &cfg, &status);
    (void)tk_mat_mul_into(&inf->xn, &lt->wk, &inf->k, &cfg, &status);
    (void)tk_mat_mul_into(&inf->xn, &lt->wv, &inf->v, &cfg, &status);
    if (status != TK_OK) return status;

    /* per-head QK norm, then RoPE */
    head_rms_norm(inf->q.data, lt->q_norm, 1, n_heads, hd, eps);
    head_rms_norm(inf->k.data, lt->k_norm, 1, n_kv, hd, eps);
    const float *cr = inf->cos_t + n * half, *sr = inf->sin_t + n * half;
    apply_rope(inf->q.data, 1, n_heads, hd, cr, sr);
    apply_rope(inf->k.data, 1, n_kv, hd, cr, sr);

    /* cache K/V, then attention using all cached keys */
    memcpy(inf->kc[l].data + n * kvwidth, inf->k.data, kvwidth * sizeof(float));
    memcpy(inf->vc[l].data + n * kvwidth, inf->v.data, kvwidth * sizeof(float));
    attention_step(inf->o.data, inf->q.data, inf->kc[l].data, inf->vc[l].data, n, n_heads, n_kv, hd, inf->prow);

    /* output projection -> residual */
    (void)tk_mat_mul_into(&inf->o, &lt->wo, &inf->ao, &cfg, &status);
    if (status != TK_OK) return status;
    for (size_t k = 0; k < hidden; k++) inf->H.data[k] += inf->ao.data[k];

    /* SwiGLU MLP -> residual */
    rms_norm(inf->H.data, lt->ffn_norm, hidden, eps, inf->xf.data);
    (void)tk_mat_mul_into(&inf->xf, &lt->wgate, &inf->gate, &cfg, &status);
    (void)tk_mat_mul_into(&inf->xf, &lt->wup, &inf->up, &cfg, &status);
    for (size_t i = 0; i < inter; i++) inf->act.data[i] = silu(inf->gate.data[i]) * inf->up.data[i];
    (void)tk_mat_mul_into(&inf->act, &lt->wdown, &inf->down, &cfg, &status);
    if (status != TK_OK) return status;
    for (size_t k = 0; k < hidden; k++) inf->H.data[k] += inf->down.data[k];
  }

  /* final norm + lm_head = tied embedding */
  rms_norm(inf->H.data, inf->out_norm_w, hidden, eps, inf->xf.data);
  TkMatrix out = {1, vocab, logits};
  (void)tk_mat_mul_into(&inf->xf, &inf->emb, &out, &cfg, &status);
  if (status != TK_OK) return status;

  inf->n = n + 1;
  return TK_OK;
}

tk_status tk_forward_logits(const TkGguf *g, const TkModel *c, const uint32_t *ids, size_t T, TkMatrix *logits) {
  TkInfer *inf = tk_infer_new(g, c);
  if (!inf) return TK_ERR_INTERNAL;
  tk_status status = TK_OK;
  *logits = tk_mat_new(T, c->vocab_size);
  float *row = tk_xmalloc(c->vocab_size * sizeof(float));
  for (size_t t = 0; t < T && status == TK_OK; t++) {
    status = tk_infer_step(inf, ids[t], row);
    if (status == TK_OK) memcpy(logits->data + t * c->vocab_size, row, c->vocab_size * sizeof(float));
  }
  free(row);
  tk_infer_free(inf);
  return status;
}
