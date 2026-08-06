#ifndef QWEN_TOKENIZER_H
#define QWEN_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

typedef struct QwenTokenizer QwenTokenizer;

/* Load the tokenizer from a Qwen GGUF file (tokenizer section in the metadata header).
 * Returns NULL on any I/O or format failure. */
QwenTokenizer *qwen_tokenizer_load(const char *gguf_path);
void qwen_tokenizer_free(QwenTokenizer *tz);

/* Tokenize null-terminated UTF-8 `text`. Sets *ids to a malloc'd array (caller frees via
 * free()) and returns the token count. Longest-match special tokens and the GGUF
 * add_bos_token flag are honored. A short text may use every byte as its own token, so
 * the returned array has at most strlen(text) + 1 elements. */
size_t qwen_tokenize(const QwenTokenizer *tz, const char *text, uint32_t **ids);

/* Detokenize ids to a malloc'd null-terminated UTF-8 string (caller frees). */
char *qwen_detokenize(const QwenTokenizer *tz, const uint32_t *ids, size_t n);

#endif /* QWEN_TOKENIZER_H */
