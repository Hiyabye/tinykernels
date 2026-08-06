#ifndef TK_GGUF_H
#define TK_GGUF_H

#include "tk_common.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Minimal GGUF (v3) reader: header + metadata KV + tensor-info table.
 *
 * Supports the value types present in Qwen GGUF models: ints, bools, floats,
 * strings, and the string/int32 arrays the tokenizer needs. Tensor writes are
 * not touched; tensor data is read on demand and dequantized to float32
 * (F32 and Q8_0). One context holds the parsed header/metadata/table; tensor
 * data stays on disk until asked for. */

typedef struct TkGguf TkGguf;

#define TK_GGUF_MAX_DIMS 4

typedef struct {
  char *name;
  uint8_t n_dims;
  uint64_t dims[TK_GGUF_MAX_DIMS]; /* ne[i]: dim0 is contiguous in memory */
  uint32_t type;                   /* ggml type id: 0=F32, 8=Q8_0 */
  uint64_t offset;                 /* bytes from the tensor_data section start */
  uint64_t n_elems;                /* product of dims */
} TkGgufTensor;

/* NULL + TK_LOGE on any I/O or format failure. */
TkGguf *tk_gguf_open(const char *path);
void tk_gguf_close(TkGguf *ctx);

/* Metadata value access (infallible: absent/non-matching returns the default). */
const char *tk_gguf_str(const TkGguf *ctx, const char *key); /* NULL if absent/non-string */
int64_t tk_gguf_i64(const TkGguf *ctx, const char *key, int64_t dflt);
double tk_gguf_f64(const TkGguf *ctx, const char *key, double dflt);
bool tk_gguf_bool(const TkGguf *ctx, const char *key, bool dflt);

/* String array (e.g. tokenizer.ggml.tokens). Returns the element count and a
 * malloc'd array of malloc'd strings in *out; caller frees each string then
 * the array. Returns 0 and leaves *out untouched (NULL) if key absent. */
size_t tk_gguf_str_array(const TkGguf *ctx, const char *key, char ***out);
/* int32 array (e.g. tokenizer.ggml.token_type). Same ownership contract. */
size_t tk_gguf_i32_array(const TkGguf *ctx, const char *key, int32_t **out);
/* Element count of a string/int32 array metadata value without copying; 0 if absent. */
size_t tk_gguf_array_len(const TkGguf *ctx, const char *key);

size_t tk_gguf_tensor_count(const TkGguf *ctx);
const TkGgufTensor *tk_gguf_tensor_index(const TkGguf *ctx, size_t i);
const TkGgufTensor *tk_gguf_tensor_named(const TkGguf *ctx, const char *name);

/* Dequantize elements [i0, i0+n) of tensor `name` into dst (float32, at least n
 * slots). Returns TK_OK on success; TK_ERR_ARG if the tensor is missing or the
 * range exceeds it, TK_ERR_UNSUPPORTED for an unknown quant type. TK_LOGE once
 * on failure. Supports F32 and Q8_0. */
tk_status tk_gguf_read(const TkGguf *ctx, const char *name, float *dst, size_t i0, size_t n);

#endif /* TK_GGUF_H */
