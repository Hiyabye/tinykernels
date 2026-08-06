#ifndef TK_MODEL_H
#define TK_MODEL_H

#include <stdbool.h>
#include <stddef.h>

struct TkGguf;

typedef struct {
  size_t hidden_size;
  size_t num_layers;
  size_t num_attention_heads;
  size_t num_kv_heads;
  size_t head_dim;
  size_t intermediate_size;
  size_t vocab_size;
  size_t max_position_embeddings;
  float rms_norm_eps;
  float rope_theta;
  bool tie_word_embeddings;
  size_t bos_token_id;
  size_t eos_token_id;
} TkModel;

TkModel tk_model_qwen3_0_6b(void);
TkModel tk_model_from_gguf(const struct TkGguf *gguf);
size_t tk_model_param_count(const TkModel *config);
void tk_model_print(const TkModel *config);

#endif /* TK_MODEL_H */
