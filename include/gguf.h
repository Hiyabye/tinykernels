#ifndef GGUF_H
#define GGUF_H

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

typedef struct GGUFContext GGUFContext;

#define GGUF_MAX_DIMS 4

typedef struct {
  char *name;
  uint8_t n_dims;
  uint64_t dims[GGUF_MAX_DIMS]; /* ne[i]: dim0 is contiguous in memory */
  uint32_t type;                /* ggml type id: 0=F32, 8=Q8_0 */
  uint64_t offset;              /* bytes from the tensor_data section start */
  uint64_t n_elems;             /* product of dims */
} GGUFTensor;

/* NULL on any I/O or format failure. */
GGUFContext *gguf_open(const char *path);
void gguf_close(GGUFContext *ctx);

/* Metadata value access. */
const char *gguf_get_string(const GGUFContext *ctx, const char *key); /* NULL if absent/non-string */
int64_t gguf_get_int(const GGUFContext *ctx, const char *key, int64_t dflt);
double gguf_get_float(const GGUFContext *ctx, const char *key, double dflt);
bool gguf_get_bool(const GGUFContext *ctx, const char *key, bool dflt);

/* String array (e.g. tokenizer.ggml.tokens). Returns the element count and a
 * malloc'd array of malloc'd strings in *out; caller frees each string then
 * the array. Returns 0 and leaves *out untouched (NULL) if key absent. */
size_t gguf_get_string_array(const GGUFContext *ctx, const char *key, char ***out);
/* int32 array (e.g. tokenizer.ggml.token_type). Same ownership contract. */
size_t gguf_get_i32_array(const GGUFContext *ctx, const char *key, int32_t **out);
/* Element count of a string/int32 array metadata value without copying; 0 if absent. */
size_t gguf_array_length(const GGUFContext *ctx, const char *key);

size_t gguf_tensor_count(const GGUFContext *ctx);
const GGUFTensor *gguf_tensor_by_index(const GGUFContext *ctx, size_t i);
const GGUFTensor *gguf_tensor_by_name(const GGUFContext *ctx, const char *name);

/* Dequantize elements [i0, i0+n) of tensor `name` into dst (float32, at least n
 * slots). Returns 0 on success, -1 if the tensor is missing or type unsupported,
 * or if i0+n exceeds the tensor. Supports F32 and Q8_0. */
int gguf_read_tensor(const GGUFContext *ctx, const char *name, float *dst, size_t i0, size_t n);

#endif /* GGUF_H */
