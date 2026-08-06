#include "tk_model.h"
#include "tk_gguf.h"

#include <stdio.h>
#include <string.h>

/* Model presets: alias -> GGUF file. Add a row here to register another model. */
static const TkModelPreset PRESETS[] = {
    {"qwen3-0.6b", "Qwen3-0.6B-Q8_0.gguf"},
};
#define PRESET_COUNT (sizeof(PRESETS) / sizeof(PRESETS[0]))

size_t tk_model_preset_count(void) { return PRESET_COUNT; }

const TkModelPreset *tk_model_preset_at(size_t i) { return i < PRESET_COUNT ? &PRESETS[i] : NULL; }

const TkModelPreset *tk_model_preset(const char *name) {
  for (size_t i = 0; i < PRESET_COUNT; i++)
    if (strcmp(PRESETS[i].name, name) == 0) return &PRESETS[i];
  return NULL;
}

const char *tk_model_default_gguf(void) { return PRESETS[0].gguf; }

TkModel tk_model_from_gguf(const struct TkGguf *g) {
  TkModel c = {0};
  size_t vocab = tk_gguf_array_len(g, "tokenizer.ggml.tokens");
  c.hidden_size = (size_t)tk_gguf_i64(g, "qwen3.embedding_length", 1024);
  c.num_layers = (size_t)tk_gguf_i64(g, "qwen3.block_count", 28);
  c.num_attention_heads = (size_t)tk_gguf_i64(g, "qwen3.attention.head_count", 16);
  c.num_kv_heads = (size_t)tk_gguf_i64(g, "qwen3.attention.head_count_kv", 8);
  c.head_dim = (size_t)tk_gguf_i64(g, "qwen3.attention.key_length", c.hidden_size / c.num_attention_heads);
  c.intermediate_size = (size_t)tk_gguf_i64(g, "qwen3.feed_forward_length", 3072);
  c.vocab_size = vocab;
  c.max_position_embeddings = (size_t)tk_gguf_i64(g, "qwen3.context_length", 40960);
  c.rms_norm_eps = (float)tk_gguf_f64(g, "qwen3.attention.layer_norm_rms_epsilon", 1e-6);
  c.rope_theta = (float)tk_gguf_f64(g, "qwen3.rope.freq_base", 1000000.0);
  c.tie_word_embeddings = true; /* Qwen3 GGUF omits output.weight; lm_head is tied */
  c.bos_token_id = (size_t)tk_gguf_i64(g, "tokenizer.ggml.bos_token_id", 151643);
  c.eos_token_id = (size_t)tk_gguf_i64(g, "tokenizer.ggml.eos_token_id", 151645);
  return c;
}

size_t tk_model_param_count(const TkModel *c) {
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

void tk_model_print(const TkModel *c) {
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
  printf("  %-25s%zu (~0.60B)\n", "param_count", tk_model_param_count(c));
}
