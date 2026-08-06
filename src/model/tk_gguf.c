#include "tk_gguf.h"
#include "tk_common.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGUF_ALIGNMENT 32 /* general.alignment absent -> default 32 */

static uint64_t le_u64(const uint8_t *p) {
  uint64_t x = 0;
  for (int i = 0; i < 8; i++) x |= ((uint64_t)p[i]) << (8 * i);
  return x;
}
static uint32_t le_u32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }

static size_t align_up(size_t n, size_t a) { return (n + a - 1) / a * a; }

/* One retained metadata KV. Types we don't need are skipped at parse time and
 * never stored. */
typedef struct {
  char *key;
  uint8_t kind; /* M_STR / M_INT / M_FLT / M_BOOL */
  char *s;
  int64_t i;
  double f;
  bool b;
  /* array values */
  uint8_t et; /* string(8) or int32(5) */
  uint64_t n;
  char **strs;
  int32_t *i32;
} MetaKV;

enum { M_STR, M_INT, M_FLT, M_BOOL };

struct TkGguf {
  FILE *fp;
  uint64_t tensor_data_start; /* byte offset of tensor_data in the file */

  MetaKV *meta;
  size_t n_meta;

  TkGgufTensor *tensors;
  size_t n_tensors;
};

/* Reads a length-prefixed GGUF string. Returns malloc'd NUL-terminated string
 * or NULL on EOF. */
static char *read_str(FILE *fp) {
  uint8_t lb[8];
  if (fread(lb, 1, 8, fp) != 8) return NULL;
  uint64_t n = le_u64(lb);
  char *s = tk_xmalloc(n + 1);
  if (fread(s, 1, n, fp) != n) {
    free(s);
    return NULL;
  }
  s[n] = '\0';
  return s;
}

static int meta_put_retry(struct TkGguf *ctx, MetaKV *m) {
  ctx->meta = tk_xrealloc(ctx->meta, (ctx->n_meta + 1) * sizeof(*ctx->meta));
  ctx->meta[ctx->n_meta++] = *m;
  return 1;
}

/* Scalar size in bytes per value type. Returns 0 for string (variable). */
static size_t scalar_size(uint32_t vt) {
  switch (vt) {
  case 0:
  case 1:
  case 7:
    return 1;
  case 2:
  case 3:
    return 2;
  case 4:
  case 5:
  case 6:
    return 4;
  case 10:
  case 11:
  case 12:
    return 8;
  default:
    return 0;
  }
}

static int64_t read_int_le(FILE *fp, size_t bytes) {
  uint8_t b[8] = {0};
  if (fread(b, 1, bytes, fp) != bytes) return 0;
  int64_t v = 0;
  for (size_t i = 0; i < bytes; i++) v |= (int64_t)b[i] << (8 * i);
  /* sign-extend */
  if (bytes < 8 && (b[bytes - 1] & 0x80)) v |= ~((1LL << (8 * bytes)) - 1);
  return v;
}
static double read_float_le(FILE *fp, size_t bytes) {
  uint8_t b[8];
  if (fread(b, 1, bytes, fp) != bytes) return 0.0;
  if (bytes == 4) {
    uint32_t u = le_u32(b);
    float f;
    memcpy(&f, &u, 4);
    return f;
  }
  uint64_t u = le_u64(b);
  double d;
  memcpy(&d, &u, 8);
  return d;
}

static bool parse_metadata(struct TkGguf *ctx, uint64_t n_kv) {
  for (uint64_t k = 0; k < n_kv; k++) {
    char *key = read_str(ctx->fp);
    if (!key) return false;

    uint8_t vtb[4];
    if (fread(vtb, 1, 4, ctx->fp) != 4) {
      free(key);
      return false;
    }
    uint32_t vt = le_u32(vtb);

    if (vt == 9) { /* array */
      uint8_t arr[12];
      if (fread(arr, 1, 12, ctx->fp) != 12) {
        free(key);
        return false;
      }
      uint32_t et = le_u32(arr);
      uint64_t cnt = le_u64(arr + 4);

      if (et == 8) { /* string array */
        char **strs = tk_xmalloc((size_t)cnt * sizeof(char *));
        bool ok = true;
        for (uint64_t c = 0; c < cnt; c++) {
          strs[c] = read_str(ctx->fp);
          if (!strs[c]) {
            ok = false;
            break;
          }
        }
        if (!ok) {
          free(strs);
          free(key);
          return false;
        }
        MetaKV m = {.key = key, .kind = M_STR, .et = 8, .n = cnt, .strs = strs};
        if (!meta_put_retry(ctx, &m)) return false;
      } else if (et == 5 || et == 4) { /* int32/uint32 array */
        int32_t *arr2 = tk_xmalloc((size_t)cnt * sizeof(int32_t));
        if (fread(arr2, 4, (size_t)cnt, ctx->fp) != (size_t)cnt) {
          free(arr2);
          free(key);
          return false;
        }
        if (et == 4) { /* reinterpret uint32 as int32 (bit-identical for our uses) */
          for (uint64_t c = 0; c < cnt; c++) arr2[c] = (int32_t)(uint32_t)arr2[c];
        }
        MetaKV m = {.key = key, .kind = M_INT, .et = 5, .n = cnt, .i32 = arr2};
        if (!meta_put_retry(ctx, &m)) return false;
      } else { /* skip unknown array: raw byte count */
        size_t esz = (et == 0 || et == 1 || et == 7) ? 1 : (et == 2 || et == 3) ? 2 : (et == 6) ? 4 : (et == 10 || et == 11 || et == 12) ? 8 : 0;
        if (esz == 0 || fseek(ctx->fp, (long)(cnt * esz), SEEK_CUR) != 0) {
          free(key);
          return false;
        }
        free(key);
      }
    } else {
      MetaKV m = {.key = key};
      if (vt == 8) { /* string scalar */
        m.kind = M_STR;
        m.s = read_str(ctx->fp);
        if (!m.s) {
          free(key);
          return false;
        }
      } else {
        size_t sz = scalar_size(vt);
        if (sz == 0) {
          free(key);
          return false;
        }
        if (vt == 6 || vt == 12) {
          m.kind = M_FLT;
          m.f = read_float_le(ctx->fp, sz);
        } else if (vt == 7) {
          uint8_t b;
          if (fread(&b, 1, 1, ctx->fp) != 1) {
            free(key);
            return false;
          }
          m.kind = M_BOOL;
          m.b = b != 0;
        } else {
          m.kind = M_INT;
          m.i = read_int_le(ctx->fp, sz);
        }
      }
      if (!meta_put_retry(ctx, &m)) return false;
    }
  }
  return true;
}

static bool parse_tensor_infos(struct TkGguf *ctx, uint64_t n_tensors) {
  ctx->tensors = tk_xmalloc((size_t)n_tensors * sizeof(TkGgufTensor));
  ctx->n_tensors = (size_t)n_tensors;

  for (uint64_t t = 0; t < n_tensors; t++) {
    TkGgufTensor *ti = &ctx->tensors[t];
    memset(ti, 0, sizeof(*ti));
    ti->name = read_str(ctx->fp);
    if (!ti->name) return false;

    uint8_t ndb[4];
    if (fread(ndb, 1, 4, ctx->fp) != 4 || (ti->n_dims = le_u32(ndb)) > TK_GGUF_MAX_DIMS) return false;

    uint8_t dimb[8 * TK_GGUF_MAX_DIMS];
    if (fread(dimb, 8, ti->n_dims, ctx->fp) != ti->n_dims) return false;
    ti->n_elems = 1;
    for (uint8_t d = 0; d < ti->n_dims; d++) {
      ti->dims[d] = (uint64_t)le_u64(dimb + 8 * d);
      ti->n_elems *= ti->dims[d];
    }

    uint8_t tb[12];
    if (fread(tb, 1, 12, ctx->fp) != 12) return false;
    ti->type = le_u32(tb);
    ti->offset = (uint64_t)le_u64(tb + 4);
  }
  return true;
}

static const MetaKV *meta_find(const TkGguf *ctx, const char *key) {
  for (size_t i = 0; i < ctx->n_meta; i++)
    if (strcmp(ctx->meta[i].key, key) == 0) return &ctx->meta[i];
  return NULL;
}

TkGguf *tk_gguf_open(const char *path) {
  FILE *fp = fopen(path, "rb");
  if (!fp) {
    TK_LOGE("cannot open GGUF file '%s': %s", path, strerror(errno));
    return NULL;
  }

  uint8_t hdr[24];
  if (fread(hdr, 1, 24, fp) != 24 || memcmp(hdr, "GGUF", 4) != 0) {
    TK_LOGE("not a valid GGUF file: '%s'", path);
    fclose(fp);
    return NULL;
  }
  uint64_t n_tensors = (uint64_t)le_u64(hdr + 8);
  uint64_t n_kv = (uint64_t)le_u64(hdr + 16);

  TkGguf *ctx = tk_xcalloc(1, sizeof(*ctx));
  ctx->fp = fp;

  if (!parse_metadata(ctx, n_kv) || !parse_tensor_infos(ctx, n_tensors)) {
    TK_LOGE("malformed GGUF metadata/tensor info in '%s'", path);
    tk_gguf_close(ctx);
    return NULL;
  }

  long pos = ftell(ctx->fp);
  if (pos < 0) {
    TK_LOGE("failed to read GGUF position in '%s'", path);
    tk_gguf_close(ctx);
    return NULL;
  }
  ctx->tensor_data_start = (uint64_t)align_up((size_t)pos, GGUF_ALIGNMENT);
  return ctx;
}

void tk_gguf_close(TkGguf *ctx) {
  if (!ctx) return;
  if (ctx->fp) fclose(ctx->fp);
  for (size_t i = 0; i < ctx->n_meta; i++) {
    free(ctx->meta[i].key);
    if (ctx->meta[i].kind == M_STR && ctx->meta[i].strs) {
      for (uint64_t j = 0; j < ctx->meta[i].n; j++) free(ctx->meta[i].strs[j]);
      free(ctx->meta[i].strs);
    } else if (ctx->meta[i].i32) {
      free(ctx->meta[i].i32);
    }
    free(ctx->meta[i].s);
  }
  free(ctx->meta);
  for (size_t i = 0; i < ctx->n_tensors; i++) free(ctx->tensors[i].name);
  free(ctx->tensors);
  free(ctx);
}

/* ---------------- metadata accessors ---------------- */

const char *tk_gguf_str(const TkGguf *ctx, const char *key) {
  const MetaKV *m = meta_find(ctx, key);
  return (m && m->kind == M_STR && m->s) ? m->s : NULL;
}

int64_t tk_gguf_i64(const TkGguf *ctx, const char *key, int64_t dflt) {
  const MetaKV *m = meta_find(ctx, key);
  return (m && m->kind == M_INT && !m->strs) ? m->i : dflt;
}

double tk_gguf_f64(const TkGguf *ctx, const char *key, double dflt) {
  const MetaKV *m = meta_find(ctx, key);
  return (m && m->kind == M_FLT) ? m->f : dflt;
}

bool tk_gguf_bool(const TkGguf *ctx, const char *key, bool dflt) {
  const MetaKV *m = meta_find(ctx, key);
  return (m && m->kind == M_BOOL) ? m->b : dflt;
}

size_t tk_gguf_str_array(const TkGguf *ctx, const char *key, char ***out) {
  const MetaKV *m = meta_find(ctx, key);
  if (!m || m->kind != M_STR || !m->strs || m->et != 8) return 0;
  char **copy = tk_xmalloc((size_t)m->n * sizeof(char *));
  for (uint64_t i = 0; i < m->n; i++) copy[i] = tk_xstrdup(m->strs[i]);
  *out = copy;
  return (size_t)m->n;
}

size_t tk_gguf_i32_array(const TkGguf *ctx, const char *key, int32_t **out) {
  const MetaKV *m = meta_find(ctx, key);
  if (!m || m->kind != M_INT || !m->i32 || m->et != 5) return 0;
  int32_t *copy = tk_xmalloc((size_t)m->n * sizeof(int32_t));
  memcpy(copy, m->i32, (size_t)m->n * sizeof(int32_t));
  *out = copy;
  return (size_t)m->n;
}

size_t tk_gguf_array_len(const TkGguf *ctx, const char *key) {
  const MetaKV *m = meta_find(ctx, key);
  return (m && m->strs) ? (size_t)m->n : 0;
}

/* ---------------- tensors ---------------- */

size_t tk_gguf_tensor_count(const TkGguf *ctx) { return ctx->n_tensors; }

const TkGgufTensor *tk_gguf_tensor_index(const TkGguf *ctx, size_t i) { return i < ctx->n_tensors ? &ctx->tensors[i] : NULL; }

const TkGgufTensor *tk_gguf_tensor_named(const TkGguf *ctx, const char *name) {
  for (size_t i = 0; i < ctx->n_tensors; i++)
    if (strcmp(ctx->tensors[i].name, name) == 0) return &ctx->tensors[i];
  return NULL;
}

/* FP16 -> float32 (bit 0x8000 sign, 5-bit exp, 10-bit mantissa). */
static float half_to_float(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x3ff;
  uint32_t f32;
  if (exp == 0) {
    if (mant == 0) {
      f32 = sign;
    } else { /* subnormal */
      int e = -1;
      uint32_t m = mant;
      do {
        e++;
        m <<= 1;
      } while (!(m & 0x400));
      f32 = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3ff) << 13);
    }
  } else if (exp == 31) { /* inf/nan */
    f32 = sign | 0x7f800000 | (mant << 13);
  } else {
    f32 = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float out;
  memcpy(&out, &f32, 4);
  return out;
}

/* Dequantize a Q8_0 block (32 elements) at file pos `pos` into y. */
static void dequant_q8_0_block(FILE *fp, long pos, float *y) {
  uint8_t b[34];
  fseek(fp, pos, SEEK_SET);
  if (fread(b, 1, 34, fp) != 34) {
    for (int i = 0; i < 32; i++) y[i] = 0.0f;
    return;
  }
  float d = half_to_float((uint16_t)b[0] | ((uint16_t)b[1] << 8));
  for (int i = 0; i < 32; i++) y[i] = d * (int8_t)b[2 + i];
}

tk_status tk_gguf_read(const TkGguf *ctx, const char *name, float *dst, size_t i0, size_t n) {
  const TkGgufTensor *ti = tk_gguf_tensor_named(ctx, name);
  if (!ti) {
    TK_LOGE("unknown tensor '%s'", name);
    return TK_ERR_ARG;
  }
  if (i0 + n > ti->n_elems) {
    TK_LOGE("tensor '%s' range [%zu,%zu) exceeds %llu elements", name, i0, i0 + n, (unsigned long long)ti->n_elems);
    return TK_ERR_ARG;
  }

  if (ti->type == 0) { /* F32 */
    if (fseek(ctx->fp, (long)(ctx->tensor_data_start + ti->offset + i0 * 4), SEEK_SET) != 0) {
      TK_LOGE("tensor '%s' seek failed", name);
      return TK_ERR_IO;
    }
    if (fread(dst, 4, n, ctx->fp) != n) {
      TK_LOGE("tensor '%s' read failed", name);
      return TK_ERR_IO;
    }
    return TK_OK;
  }

  if (ti->type == 8) { /* Q8_0 */
    float block[32];
    size_t done = 0;
    while (done < n) {
      size_t idx = i0 + done;
      size_t blk = idx / 32, within = idx % 32;
      long pos = (long)(ctx->tensor_data_start + ti->offset + blk * 34);
      dequant_q8_0_block(ctx->fp, pos, block);
      size_t take = 32 - within;
      if (take > n - done) take = n - done;
      memcpy(dst + done, block + within, take * sizeof(float));
      done += take;
    }
    return TK_OK;
  }

  TK_LOGE("tensor '%s' has unsupported quant type %u", name, ti->type);
  return TK_ERR_UNSUPPORTED;
}
