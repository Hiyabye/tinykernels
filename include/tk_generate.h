#ifndef TK_GENERATE_H
#define TK_GENERATE_H

#include <stddef.h>
#include <stdint.h>

struct TkTokenizer;

/* Build the Qwen3 chat-template ids for a single user turn. When `think` is
 * nonzero the assistant turn is prefixed with the ` thinking` token so the
 * model emits a reasoning block before ` response`. Returns a malloc'd id
 * array (caller frees) and sets *n to its length. */
uint32_t *tk_chat_prompt(const struct TkTokenizer *tz, const char *user, int think, size_t *n);

/* Sample one token index from a logits row. temperature <= 0 selects greedy.
 * top_k <= 0 disables top-k; top_p in (0,1) enables nucleus filtering. `rng`
 * is a mutable xorshift64* state (seeded once by the caller); identical
 * {logits, params, seed} inputs give identical draws. */
uint32_t tk_sample(const float *logits, size_t vocab, float temperature, int top_k, float top_p, uint64_t *rng);

#endif /* TK_GENERATE_H */
