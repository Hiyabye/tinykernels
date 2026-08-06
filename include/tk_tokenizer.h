#ifndef TK_TOKENIZER_H
#define TK_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

typedef struct TkTokenizer TkTokenizer;

/* Load the tokenizer from a Qwen GGUF file (tokenizer section in the metadata header).
 * Returns NULL on any I/O or format failure (TK_LOGE already emitted). */
TkTokenizer *tk_tokenizer_open(const char *gguf_path);
void tk_tokenizer_close(TkTokenizer *tz);

/* Tokenize null-terminated UTF-8 `text`. Sets *ids to a malloc'd array (caller frees via
 * free()) and returns the token count. Longest-match special tokens and the GGUF
 * add_bos_token flag are honored. A short text may use every byte as its own token, so
 * the returned array has at most strlen(text) + 1 elements. */
size_t tk_tokenize(const TkTokenizer *tz, const char *text, uint32_t **ids);

/* Detokenize ids to a malloc'd null-terminated UTF-8 string (caller frees). Out-of-range
 * ids (>= vocab_size) are skipped. */
char *tk_detokenize(const TkTokenizer *tz, const uint32_t *ids, size_t n);

#endif /* TK_TOKENIZER_H */
