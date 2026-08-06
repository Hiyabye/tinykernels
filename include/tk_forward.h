#ifndef TK_FORWARD_H
#define TK_FORWARD_H

#include "tk_common.h"
#include "tk_gguf.h"
#include "tk_matrix.h"
#include "tk_model.h"

#include <stddef.h>
#include <stdint.h>

/* Run the full Qwen3 forward pass for a prompt of `n_tokens` ide, producing one
 * logits row per token (rows = n_tokens, cols = vocab_size). Weights are
 * dequantized on demand from `gguf`. On success *logits is a freshly allocated
 * matrix the caller frees with tk_mat_free. Returns the first failing tk_status
 * (TK_LOGE already emitted by the gguf layer), else TK_OK. */
tk_status tk_forward_logits(const TkGguf *gguf, const TkModel *config, const uint32_t *ids, size_t n_tokens, TkMatrix *logits);

#endif /* TK_FORWARD_H */
