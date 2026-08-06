#ifndef QWEN_H
#define QWEN_H

#include <stdbool.h>
#include <stddef.h>

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
} QwenConfig;

QwenConfig qwen_config_qwen3_0_6b(void);
size_t qwen_config_param_count(const QwenConfig *config);
void qwen_config_print(const QwenConfig *config);

#endif // QWEN_H
