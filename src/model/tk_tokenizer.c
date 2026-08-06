#include "tk_tokenizer.h"
#include "internal/unicode_ranges.h"
#include "tk_gguf.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Byte-level BPE helpers
 * ========================================================================= */

static uint32_t utf8_cp(const char *s, size_t *len) {
  const uint8_t *u = (const uint8_t *)s;
  if (u[0] < 0x80) {
    *len = 1;
    return u[0];
  }
  if ((u[0] & 0xE0) == 0xC0) {
    *len = 2;
    return ((uint32_t)(u[0] & 0x1F) << 6) | (u[1] & 0x3F);
  }
  if ((u[0] & 0xF0) == 0xE0) {
    *len = 3;
    return ((uint32_t)(u[0] & 0x0F) << 12) | ((uint32_t)(u[1] & 0x3F) << 6) | (u[2] & 0x3F);
  }
  if ((u[0] & 0xF8) == 0xF0) {
    *len = 4;
    return ((uint32_t)(u[0] & 0x07) << 18) | ((uint32_t)(u[1] & 0x3F) << 12) | ((uint32_t)(u[2] & 0x3F) << 6) | (u[3] & 0x3F);
  }
  *len = 0;
  return 0;
}

/* Hash map: keepable-token string -> id (open addressing, linear probing). */
typedef struct {
  char *key;
  size_t val;
  bool used;
} Entry;
static uint32_t hash_str(const char *s) {
  uint32_t h = 2166136261u;
  for (const uint8_t *p = (const uint8_t *)s; *p; p++) {
    h ^= *p;
    h *= 16777619u;
  }
  return h;
}
static void smap_put(Entry *map, size_t cap, const char *key, size_t val) {
  size_t idx = hash_str(key) & (cap - 1);
  while (map[idx].used) idx = (idx + 1) & (cap - 1);
  map[idx].key = (char *)key;
  map[idx].val = val;
  map[idx].used = true;
}
static size_t smap_get(const Entry *map, size_t cap, const char *key) {
  size_t idx = hash_str(key) & (cap - 1);
  while (map[idx].used) {
    if (strcmp(map[idx].key, key) == 0) return map[idx].val;
    idx = (idx + 1) & (cap - 1);
  }
  return SIZE_MAX;
}

struct TkTokenizer {
  size_t vocab_size;
  char **tokens;    /* id -> mapped-space string                        */
  uint8_t *special; /* id -> 1 if a non-mergeable special token         */
  size_t num_special;
  uint32_t *special_ids;
  bool add_bos_token;
  uint32_t bos_token_id;

  uint32_t byte_cp[256];     /* byte -> mapped codepoint                       */
  uint32_t uni_to_byte[512]; /* mapped codepoint -> byte                      */
  uint16_t byte_to_id[256];  /* byte -> byte-token id                         */

  Entry *map;
  size_t map_cap;
};

/* Unicode category classification matching llama.cpp's unicode_cpt_flags:
 *   is_letter(cp) = general category L*   (\p{L})
 *   is_number(cp) = general category N*   (\p{N})
 *   is_whitespace  = llama.cpp White_Space set.
 * The letter/number tables are generated (include/internal/unicode_ranges.h),
 * see scripts/gen-unicode-data.py equivalent; whitespace is the explicit set. */
static bool in_ranges(uint32_t cp, const qwen_unicode_range *r, size_t n) {
  size_t lo = 0, hi = n;
  while (lo < hi) {
    size_t m = lo + (hi - lo) / 2;
    if (cp > r[m].hi) lo = m + 1;
    else hi = m;
  }
  return lo < n && cp >= r[lo].lo;
}
static bool is_letter_cp(uint32_t cp) { return in_ranges(cp, qwen_letter_ranges, QWEN_LETTER_COUNT); }
static bool is_number_cp(uint32_t cp) { return in_ranges(cp, qwen_number_ranges, QWEN_NUMBER_COUNT); }
static bool is_ws_cp(uint32_t cp) {
  return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 ||
         cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}
static uint32_t to_lower_ascii(uint32_t cp) { return (cp >= 'A' && cp <= 'Z') ? cp + 0x20 : cp; }

/* Re-encode a codepoint to UTF-8; returns byte count. */
static size_t utf8_enc(uint32_t cp, uint8_t *out) {
  if (cp < 0x80) {
    out[0] = (uint8_t)cp;
    return 1;
  }
  if (cp < 0x800) {
    out[0] = (uint8_t)(0xC0 | (cp >> 6));
    out[1] = (uint8_t)(0x80 | (cp & 0x3F));
    return 2;
  }
  if (cp < 0x10000) {
    out[0] = (uint8_t)(0xE0 | (cp >> 12));
    out[1] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (uint8_t)(0x80 | (cp & 0x3F));
    return 3;
  }
  out[0] = (uint8_t)(0xF0 | (cp >> 18));
  out[1] = (uint8_t)(0x80 | ((cp >> 12) & 0x3F));
  out[2] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
  out[3] = (uint8_t)(0x80 | (cp & 0x3F));
  return 4;
}

/* Qwen2 GPT-2-style pre-tokenization over raw unicode codepoints.
 * Ported from llama.cpp unicode_regex_split_custom_qwen2. Writes the exclusive
 * end index of each word into `ends` and returns the word count. */
static size_t qwen2_split_words(const uint32_t *c, size_t n, size_t *ends) {
  size_t nw = 0, prev = 0, pos = 0;
  while (pos < n) {
    uint32_t cp = c[pos];
    /* (?i:'s|'t|'re|'ve|'m|'ll|'d) */
    if (cp == '\'' && pos + 1 < n) {
      uint32_t nx = to_lower_ascii(c[pos + 1]);
      if (nx == 's' || nx == 't' || nx == 'm' || nx == 'd') {
        pos += 2;
        ends[nw++] = pos;
        prev = pos;
        continue;
      }
      if (pos + 2 < n) {
        uint32_t nn = to_lower_ascii(c[pos + 2]);
        if ((nx == 'r' && nn == 'e') || (nx == 'v' && nn == 'e') || (nx == 'l' && nn == 'l')) {
          pos += 3;
          ends[nw++] = pos;
          prev = pos;
          continue;
        }
      }
    }
    /* [^\r\n\p{L}\p{N}]?\p{L}+ */
    if (!(cp == '\r' || cp == '\n' || is_number_cp(cp))) {
      if (is_letter_cp(cp) || (pos + 1 < n && is_letter_cp(c[pos + 1]))) {
        pos++;
        while (pos < n && is_letter_cp(c[pos])) pos++;
        ends[nw++] = pos;
        prev = pos;
        continue;
      }
    }
    /* \p{N} */
    if (is_number_cp(cp)) {
      pos++;
      ends[nw++] = pos;
      prev = pos;
      continue;
    }
    /* <space>?[^\s\p{L}\p{N}]+[\r\n]* */
    {
      size_t wp = (cp == ' ') ? pos + 1 : pos;
      bool w = is_ws_cp(wp < n ? c[wp] : 0), l = is_letter_cp(wp < n ? c[wp] : 0), k = is_number_cp(wp < n ? c[wp] : 0);
      if (!(w || l || k)) {
        pos += (cp == ' ');
        while (pos < n) {
          if (is_ws_cp(c[pos]) || is_letter_cp(c[pos]) || is_number_cp(c[pos])) break;
          pos++;
        }
        while (pos < n && (c[pos] == '\r' || c[pos] == '\n')) pos++;
        ends[nw++] = pos;
        prev = pos;
        continue;
      }
    }
    /* whitespace run */
    size_t num_ws = 0, last_rn = 0;
    while (pos + num_ws < n && is_ws_cp(c[pos + num_ws])) {
      uint32_t c2 = c[pos + num_ws];
      if (c2 == '\r' || c2 == '\n') last_rn = pos + num_ws + 1;
      num_ws++;
    }
    /* \s*[\r\n]+ */
    if (last_rn > 0) {
      pos = last_rn;
      ends[nw++] = pos;
      prev = pos;
      continue;
    }
    /* \s+(?!\S) */
    if (num_ws > 1 && pos + num_ws < n) {
      pos += num_ws - 1;
      ends[nw++] = pos;
      prev = pos;
      continue;
    }
    /* \s+ */
    if (num_ws > 0) {
      pos += num_ws;
      ends[nw++] = pos;
      prev = pos;
      continue;
    }
    pos++;
    ends[nw++] = pos;
    prev = pos;
  }
  (void)prev;
  return nw;
}

typedef struct {
  uint32_t *v;
  size_t n, cap;
} U32Buf;
static void ubuf_push(U32Buf *b, uint32_t x) {
  if (b->n == b->cap) {
    b->cap = b->cap ? b->cap * 2 : 64;
    b->v = tk_xrealloc(b->v, b->cap * sizeof(uint32_t));
  }
  b->v[b->n++] = x;
}

static size_t merge_rank(const TkTokenizer *tz, uint32_t a, uint32_t b) {
  const char *sa = tz->tokens[a], *sb = tz->tokens[b];
  size_t la = strlen(sa), lb = strlen(sb);
  char tmp[la + lb + 1];
  memcpy(tmp, sa, la);
  memcpy(tmp + la, sb, lb);
  tmp[la + lb] = '\0';
  return smap_get(tz->map, tz->map_cap, tmp);
}

/* Greedy BPE on a word of byte-token ids; appends the merged ids to `out`. */
static void bpe_run(const TkTokenizer *tz, const uint32_t *word_ids, size_t n, U32Buf *out) {
  if (n == 1) {
    ubuf_push(out, word_ids[0]);
    return;
  }
  size_t wn = n;
  uint32_t *word = tk_xmalloc(n * sizeof(uint32_t));
  memcpy(word, word_ids, n * sizeof(uint32_t));
  for (;;) {
    size_t best = SIZE_MAX, rank = SIZE_MAX;
    for (size_t i = 0; i + 1 < wn; i++) {
      size_t r = merge_rank(tz, word[i], word[i + 1]);
      if (r < rank) {
        rank = r;
        best = i;
      }
    }
    if (best == SIZE_MAX) break;
    uint32_t a = word[best], merged = (uint32_t)rank;
    uint32_t *nw = tk_xmalloc(wn * sizeof(uint32_t));
    size_t nn = 0, i = 0;
    while (i < wn) {
      if (i + 1 < wn && word[i] == a && word[i + 1] == word[best + 1]) {
        nw[nn++] = merged;
        i += 2;
      } else nw[nn++] = word[i++];
    }
    free(word);
    word = nw;
    wn = nn;
    if (wn <= 1) break;
  }
  for (size_t i = 0; i < wn; i++) ubuf_push(out, word[i]);
  free(word);
}

/* BPE a single word [start, end) of codepoints: re-encode to bytes, map each
 * byte to its byte-token id, then run greedy BPE. */
static void bpe_word(const TkTokenizer *tz, const uint32_t *c, size_t start, size_t end, U32Buf *out) {
  size_t blen = 0;
  for (size_t i = start; i < end; i++) blen += (c[i] < 0x80) ? 1 : (c[i] < 0x800) ? 2 : (c[i] < 0x10000) ? 3 : 4;
  uint32_t *wids = tk_xmalloc(blen * sizeof(uint32_t));
  size_t wn = 0;
  uint8_t buf[4];
  for (size_t i = start; i < end; i++) {
    size_t nb = utf8_enc(c[i], buf);
    for (size_t j = 0; j < nb; j++) wids[wn++] = tz->byte_to_id[buf[j]];
  }
  bpe_run(tz, wids, wn, out);
  free(wids);
}

/* BPE a contiguous byte region that contains no special token strings. */
static void bpe_segment(const TkTokenizer *tz, const uint8_t *bytes, size_t n, U32Buf *out) {
  if (n == 0) return;

  /* Decode raw UTF-8 to codepoints (invalid bytes kept as-is, 1 byte each). */
  uint32_t *c = tk_xmalloc(n * sizeof(uint32_t));
  size_t nc = 0, i = 0;
  while (i < n) {
    size_t cl;
    uint32_t cp = utf8_cp((const char *)bytes + i, &cl);
    if (cl == 0) {
      c[nc++] = bytes[i];
      i++;
    } else {
      c[nc++] = cp;
      i += cl;
    }
  }

  size_t *ends = tk_xmalloc(nc * sizeof(size_t));
  size_t nw = qwen2_split_words(c, nc, ends);
  size_t start = 0;
  for (size_t k = 0; k < nw; k++) {
    bpe_word(tz, c, start, ends[k], out);
    start = ends[k];
  }
  if (start < nc) bpe_word(tz, c, start, nc, out); /* defensive; split covers all */
  free(c);
  free(ends);
}

/* Longest-match special token at byte position i, or UINT32_MAX if none. */
static uint32_t find_special(const TkTokenizer *tz, const uint8_t *bytes, size_t i, size_t len, size_t *m) {
  size_t best_len = 0;
  uint32_t best = UINT32_MAX;
  for (size_t s = 0; s < tz->num_special; s++) {
    const char *sp = tz->tokens[tz->special_ids[s]];
    size_t sl = strlen(sp);
    if (sl > best_len && i + sl <= len && memcmp(bytes + i, sp, sl) == 0) {
      best_len = sl;
      best = tz->special_ids[s];
    }
  }
  if (m) *m = best_len;
  return best;
}

/* =========================================================================
 * Public API
 * ========================================================================= */

TkTokenizer *tk_tokenizer_open(const char *path) {
  TkGguf *ctx = tk_gguf_open(path);
  if (!ctx) return NULL;

  size_t ntok;
  char **tokens = NULL;
  if ((ntok = tk_gguf_str_array(ctx, "tokenizer.ggml.tokens", &tokens)) == 0) {
    TK_LOGE("tokenizer: missing tokenizer.ggml.tokens in '%s'", path);
    tk_gguf_close(ctx);
    return NULL;
  }

  /* token_type is optional (defaults to NORMAL). It comes back as int32(). */
  int32_t *token_types = NULL;
  if (tk_gguf_i32_array(ctx, "tokenizer.ggml.token_type", &token_types) != ntok) {
    free(token_types);
    token_types = NULL;
  }

  TkTokenizer *tz = tk_xcalloc(1, sizeof(*tz));
  tz->vocab_size = ntok;
  tz->tokens = tokens;
  tz->add_bos_token = tk_gguf_bool(ctx, "tokenizer.ggml.add_bos_token", false);
  tz->bos_token_id = (uint32_t)tk_gguf_i64(ctx, "tokenizer.ggml.bos_token_id", 0);
  tz->special = tk_xcalloc(ntok, 1);

  /* Non-mergeable = everything that isn't NORMAL(1) or BYTE(6). Matched verbatim
   * (longest match) and never BPE-merged. UNUSED(5) pad tokens don't occur in real
   * input, so marking them special is harmless. */
  size_t ns = 0;
  for (size_t id = 0; id < ntok; id++) {
    int32_t tt = token_types ? token_types[id] : 1;
    if (tt != 1 && tt != 6) {
      tz->special[id] = 1;
      ns++;
    }
  }
  tz->num_special = ns;
  tz->special_ids = tk_xmalloc(ns * sizeof(uint32_t));
  for (size_t id = 0, s = 0; id < ntok; id++)
    if (tz->special[id]) tz->special_ids[s++] = (uint32_t)id;

  /* String->id map over keepable tokens (byte tokens + merges). */
  size_t keepable = ntok - ns;
  size_t cap = 1;
  while (cap < keepable * 2) cap <<= 1;
  tz->map_cap = cap;
  tz->map = tk_xcalloc(cap, sizeof(Entry));
  for (size_t id = 0; id < ntok; id++)
    if (!tz->special[id]) smap_put(tz->map, cap, tz->tokens[id], id);

  /* byte <-> unicode tables (GPT-2 bytes_to_unicode). */
  uint8_t kept[256] = {0};
  for (int b = '!'; b <= '~'; b++) kept[b] = 1;
  for (int b = 0xA1; b <= 0xAC; b++) kept[b] = 1;
  for (int b = 0xAE; b <= 0xFF; b++) kept[b] = 1;
  for (int b = 0, n = 0; b < 256; b++) tz->byte_cp[b] = kept[b] ? (uint32_t)b : 256u + n++;
  for (int b = 0; b < 256; b++) tz->uni_to_byte[tz->byte_cp[b]] = (uint32_t)b;
  for (size_t id = 0; id < 256 && id < ntok; id++) {
    size_t cl;
    uint32_t cp = utf8_cp(tz->tokens[id], &cl);
    tz->byte_to_id[tz->uni_to_byte[cp]] = (uint16_t)id;
  }

  free(token_types);
  tk_gguf_close(ctx);
  return tz;
}

void tk_tokenizer_close(TkTokenizer *tz) {
  if (!tz) return;
  for (size_t i = 0; i < tz->vocab_size; i++) free(tz->tokens[i]);
  free(tz->tokens);
  free(tz->special);
  free(tz->special_ids);
  free(tz->map);
  free(tz);
}

size_t tk_tokenize(const TkTokenizer *tz, const char *text, uint32_t **out_ids) {
  const uint8_t *bytes = (const uint8_t *)text;
  size_t len = strlen(text);
  U32Buf out = {0};

  size_t i = 0;
  while (i < len) {
    size_t sp_len;
    uint32_t sp = find_special(tz, bytes, i, len, &sp_len);
    if (sp != UINT32_MAX) {
      ubuf_push(&out, sp);
      i += sp_len;
      continue;
    }
    size_t start = i;
    while (i < len && find_special(tz, bytes, i, len, NULL) == UINT32_MAX) i++;
    bpe_segment(tz, bytes + start, i - start, &out);
  }

  *out_ids = out.v;
  if (tz->add_bos_token) { /* prepend BOS (not the case for Qwen3-0.6B, add_bos=false) */
    uint32_t *tmp = tk_xmalloc((out.n + 1) * sizeof(uint32_t));
    tmp[0] = tz->bos_token_id;
    memcpy(tmp + 1, out.v, out.n * sizeof(uint32_t));
    free(out.v);
    out.v = tmp;
    out.n++;
  }
  return out.n;
}

uint32_t tk_token_id(const TkTokenizer *tz, const char *s) {
  for (size_t id = 0; id < tz->vocab_size; id++)
    if (strcmp(tz->tokens[id], s) == 0) return (uint32_t)id;
  return UINT32_MAX;
}

char *tk_detokenize(const TkTokenizer *tz, const uint32_t *ids, size_t n) {
  size_t total = 0;
  for (size_t k = 0; k < n; k++) {
    if (ids[k] >= tz->vocab_size) continue; /* skip invalid ids */
    const char *t = tz->tokens[ids[k]];
    if (tz->special[ids[k]]) {
      total += strlen(t);
      continue;
    }
    for (const char *p = t; *p;) {
      size_t cl;
      utf8_cp(p, &cl);
      if (cl == 0) {
        total += 1;
        p++;
      } else {
        total += 1;
        p += cl;
      }
    }
  }
  char *out = tk_xmalloc(total + 1);
  size_t o = 0;
  for (size_t k = 0; k < n; k++) {
    if (ids[k] >= tz->vocab_size) continue;
    const char *t = tz->tokens[ids[k]];
    if (tz->special[ids[k]]) {
      for (const char *p = t; *p; p++) out[o++] = *p;
      continue;
    }
    for (const char *p = t; *p;) {
      size_t cl;
      uint32_t cp = utf8_cp(p, &cl);
      if (cl == 0) {
        out[o++] = *p;
        p++;
      } else {
        out[o++] = (char)tz->uni_to_byte[cp];
        p += cl;
      }
    }
  }
  out[o] = '\0';
  return out;
}
