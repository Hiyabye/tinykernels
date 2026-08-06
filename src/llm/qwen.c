#include "qwen.h"

#include <stdio.h>

QwenConfig qwen_config_qwen3_0_6b(void) {
  return (QwenConfig){
      .hidden_size = 1024,
      .num_layers = 28,
      .num_attention_heads = 16,
      .num_kv_heads = 8,
      .head_dim = 128,
      .intermediate_size = 3072,
      .vocab_size = 151936,
      .max_position_embeddings = 40960,
      .rms_norm_eps = 1e-6f,
      .rope_theta = 1000000.0f,
      .tie_word_embeddings = true,
      .bos_token_id = 151643,
      .eos_token_id = 151645,
  };
}

size_t qwen_config_param_count(const QwenConfig *c) {
  size_t head_total = c->num_attention_heads * c->head_dim;      // Wq/Wo rows
  size_t kv_total = c->num_kv_heads * c->head_dim;               // Wk/Wv rows
  size_t per_layer = c->hidden_size * head_total                 // Wq
                     + c->hidden_size * kv_total                 // Wk
                     + c->hidden_size * kv_total                 // Wv
                     + head_total * c->hidden_size               // Wo
                     + 3 * c->hidden_size * c->intermediate_size // gate/up/down
                     + 2 * c->hidden_size;                       // two RMSNorm vectors

  size_t total = c->vocab_size * c->hidden_size; // embedding (tied lm_head)
  total += c->num_layers * per_layer;
  total += c->hidden_size; // final RMSNorm
  if (!c->tie_word_embeddings) total += c->vocab_size * c->hidden_size;
  return total;
}

void qwen_config_print(const QwenConfig *c) {
  printf("Qwen3-0.6B (Qwen3ForCausalLM, causal LM)\n");
  printf("  %-25s%zu\n", "hidden_size", c->hidden_size);
  printf("  %-25s%zu\n", "num_layers", c->num_layers);
  printf("  %-25s%zu\n", "num_attention_heads", c->num_attention_heads);
  printf("  %-25s%zu\n", "num_kv_heads", c->num_kv_heads);
  printf("  %-25s%zu\n", "head_dim", c->head_dim);
  printf("  %-25s%zu\n", "intermediate_size", c->intermediate_size);
  printf("  %-25s%zu\n", "vocab_size", c->vocab_size);
  printf("  %-25s%zu\n", "max_position_embeddings", c->max_position_embeddings);
  printf("  %-25s%g\n", "rms_norm_eps", c->rms_norm_eps);
  printf("  %-25s%g\n", "rope_theta", c->rope_theta);
  printf("  %-25s%s\n", "tie_word_embeddings", c->tie_word_embeddings ? "true" : "false");
  printf("  %-25s%zu\n", "bos_token_id", c->bos_token_id);
  printf("  %-25s%zu\n", "eos_token_id", c->eos_token_id);
  printf("  %-25s%zu (~0.60B)\n", "param_count", qwen_config_param_count(c));
}
