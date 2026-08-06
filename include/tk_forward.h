#ifndef TK_FORWARD_H
#define TK_FORWARD_H

#include "tk_backend.h"
#include "tk_common.h"
#include "tk_gguf.h"
#include "tk_matrix.h"
#include "tk_model.h"

#include <stddef.h>
#include <stdint.h>

/* Incremental forward engine with a per-layer KV cache, the compute core for
 * autoregressive generation. */
typedef struct TkInfer TkInfer;

/* Allocate an inference engine bound to `gguf`/`config`. Loads the tied
 * embedding (lm_head) resident and allocates the KV cache (which grows on
 * demand). Returns NULL + TK_LOGE (from the gguf layer) on load failure. */
TkInfer *tk_infer_new(const TkGguf *gguf, const TkModel *config);
void tk_infer_free(TkInfer *inf);

/* Number of tokens processed so far (= filled KV rows). */
size_t tk_infer_len(const TkInfer *inf);

/* Override the GEMM config used for the QKV/MLP/logits matmuls. Defaults to
 * single-threaded SIMD IKJ (the fastest GEMV-shaped config for this model). */
void tk_infer_set_cfg(TkInfer *inf, const tk_matmul_cfg *config);

/* Run the forward pass for one `id` at the current position and write its
 * logits row (vocab_size floats) into `logits`. The Q/K/V for this position
 * are cached for reuse by subsequent steps. Returns the first failing
 * tk_status (the gguf layer already emitted TK_LOGE). */
tk_status tk_infer_step(TkInfer *inf, uint32_t id, float *logits);

/* Run the full forward pass for a prompt of `n_tokens` ids, producing one
 * logits row per token (rows = n_tokens, cols = vocab_size). On success
 * *logits is a freshly allocated matrix the caller frees with tk_mat_free.
 * Returns the first failing tk_status, else TK_OK. */
tk_status tk_forward_logits(const TkGguf *gguf, const TkModel *config, const uint32_t *ids, size_t n_tokens, TkMatrix *logits);

#endif /* TK_FORWARD_H */
