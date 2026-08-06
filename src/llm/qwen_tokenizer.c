#include "qwen_tokenizer.h"
#include "unicode_ranges.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * GGUF metadata reading (tokenizer section only)
 *
 * Reads only the header + the metadata KV entries needed for the tokenizer.
 * A full GGUF loader (header + tensor infos + weights) arrives in Phase 3 and
 * will absorb or replace this.
 * ========================================================================= */

typedef struct {
  char **tokens; /* vocab strings, in byte->unicode "mapped" space  */
  size_t ntokens;
  uint32_t *token_types; /* per-token GGUF type                             */
  bool add_bos_token;
  uint32_t bos_token_id;
} TZParsed;

static uint32_t le_u32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint64_t le_u64(const uint8_t *p) {
  uint64_t x = 0;
  for (int i = 0; i < 8; i++) x |= ((uint64_t)p[i]) << (8 * i);
  return x;
}

/* Reads a length-prefixed GGUF string (null-terminated) or NULL on EOF. */
static char *read_str(FILE *fp) {
  uint8_t lb[8];
  if (fread(lb, 1, 8, fp) != 8) return NULL;
  uint64_t n = le_u64(lb);
  char *s = malloc(n + 1);
  if (!s) return NULL;
  if (fread(s, 1, n, fp) != n) {
    free(s);
    return NULL;
  }
  s[n] = '\0';
  return s;
}

static bool tz_parse(const char *path, TZParsed *out) {
  memset(out, 0, sizeof(*out));
  FILE *fp = fopen(path, "rb");
  if (!fp) return false;

  uint8_t hdr[24];
  if (fread(hdr, 1, 24, fp) != 24 || memcmp(hdr, "GGUF", 4) != 0) {
    fclose(fp);
    return false;
  }
  uint64_t kv_count = le_u64(hdr + 16);

  for (uint64_t k = 0; k < kv_count; k++) {
    char *key = read_str(fp);
    if (!key) {
      fclose(fp);
      return false;
    }
    uint8_t vtb[4];
    bool ok = true;
    if (fread(vtb, 1, 4, fp) != 4) ok = false;
    uint32_t vt = ok ? le_u32(vtb) : 0;

    if (ok && vt == 9) { /* array */
      uint8_t arr[12];
      if (fread(arr, 1, 12, fp) != 12) ok = false;
      uint32_t et = ok ? le_u32(arr) : 0;
      uint64_t cnt = ok ? le_u64(arr + 4) : 0;
      if (ok && strcmp(key, "tokenizer.ggml.tokens") == 0) {
        out->ntokens = (size_t)cnt;
        out->tokens = malloc((size_t)cnt * sizeof(char *));
        if (!out->tokens) ok = false;
        for (uint64_t c = 0; ok && c < cnt; c++) {
          out->tokens[c] = read_str(fp);
          if (!out->tokens[c]) ok = false;
        }
      } else if (ok && strcmp(key, "tokenizer.ggml.token_type") == 0) {
        out->token_types = malloc((size_t)cnt * sizeof(uint32_t));
        if (!out->token_types || fread(out->token_types, 4, (size_t)cnt, fp) != (size_t)cnt) ok = false;
      } else if (ok && et == 8) { /* string array to skip */
        for (uint64_t c = 0; c < cnt; c++) {
          char *s = read_str(fp);
          if (!s) {
            ok = false;
            break;
          }
          free(s);
        }
      } else { /* unknown array, skip raw */
        uint8_t esz = (et == 0 || et == 1 || et == 7)      ? 1
                      : (et == 2 || et == 3)               ? 2
                      : (et >= 4 && et <= 6)               ? 4
                      : (et == 10 || et == 11 || et == 12) ? 8
                                                           : 0;
        if (fseek(fp, (long)(cnt * esz), SEEK_CUR) != 0) ok = false;
      }
    } else if (ok) { /* scalar */
      switch (vt) {
      case 4: { /* u32 */
        if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0) {
          uint8_t b[4];
          if (fread(b, 1, 4, fp) != 4) ok = false;
          else out->bos_token_id = le_u32(b);
        } else if (fseek(fp, 4, SEEK_CUR) != 0) ok = false;
        break;
      }
      case 7: { /* bool */
        if (strcmp(key, "tokenizer.ggml.add_bos_token") == 0) {
          uint8_t b;
          if (fread(&b, 1, 1, fp) != 1) ok = false;
          else out->add_bos_token = b != 0;
        } else if (fseek(fp, 1, SEEK_CUR) != 0) ok = false;
        break;
      }
      case 8: {
        char *s = read_str(fp);
        if (!s) ok = false;
        else free(s);
        break;
      }
      case 0:
      case 1: {
        ok = fseek(fp, 1, SEEK_CUR) == 0;
        break;
      }
      case 2:
      case 3: {
        ok = fseek(fp, 2, SEEK_CUR) == 0;
        break;
      }
      case 5:
      case 6: {
        ok = fseek(fp, 4, SEEK_CUR) == 0;
        break;
      }
      case 10:
      case 11:
      case 12: {
        ok = fseek(fp, 8, SEEK_CUR) == 0;
        break;
      }
      default:
        ok = false;
      }
    }

    free(key);
    if (!ok) {
      fclose(fp);
      return false;
    }
  }
  fclose(fp);
  return true;
}

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

struct QwenTokenizer {
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
 * The letter/number tables are generated (src/llm/unicode_ranges.h), see
 * scripts/gen-unicode-data.py equivalent; whitespace is the explicit set. */
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
    b->v = realloc(b->v, b->cap * sizeof(uint32_t));
  }
  b->v[b->n++] = x;
}

static size_t merge_rank(const QwenTokenizer *tz, uint32_t a, uint32_t b) {
  const char *sa = tz->tokens[a], *sb = tz->tokens[b];
  size_t la = strlen(sa), lb = strlen(sb);
  char tmp[la + lb + 1];
  memcpy(tmp, sa, la);
  memcpy(tmp + la, sb, lb);
  tmp[la + lb] = '\0';
  return smap_get(tz->map, tz->map_cap, tmp);
}

/* Greedy BPE on a word of byte-token ids; appends the merged ids to `out`. */
static void bpe_run(const QwenTokenizer *tz, const uint32_t *word_ids, size_t n, U32Buf *out) {
  if (n == 1) {
    ubuf_push(out, word_ids[0]);
    return;
  }
  size_t wn = n;
  uint32_t *word = malloc(n * sizeof(uint32_t));
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
    uint32_t *nw = malloc(wn * sizeof(uint32_t));
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
static void bpe_word(const QwenTokenizer *tz, const uint32_t *c, size_t start, size_t end, U32Buf *out) {
  size_t blen = 0;
  for (size_t i = start; i < end; i++) blen += (c[i] < 0x80) ? 1 : (c[i] < 0x800) ? 2 : (c[i] < 0x10000) ? 3 : 4;
  uint32_t *wids = malloc(blen * sizeof(uint32_t));
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
static void bpe_segment(const QwenTokenizer *tz, const uint8_t *bytes, size_t n, U32Buf *out) {
  if (n == 0) return;

  /* Decode raw UTF-8 to codepoints (invalid bytes kept as-is, 1 byte each). */
  uint32_t *c = malloc(n * sizeof(uint32_t));
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

  size_t *ends = malloc(nc * sizeof(size_t));
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
static uint32_t find_special(const QwenTokenizer *tz, const uint8_t *bytes, size_t i, size_t len, size_t *m) {
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

QwenTokenizer *qwen_tokenizer_load(const char *path) {
  TZParsed p;
  if (!tz_parse(path, &p)) return NULL;

  QwenTokenizer *tz = calloc(1, sizeof(*tz));
  tz->vocab_size = p.ntokens;
  tz->tokens = p.tokens;
  tz->add_bos_token = p.add_bos_token;
  tz->bos_token_id = p.bos_token_id;
  tz->special = calloc(p.ntokens, 1);

  /* Special = CONTROL(3) or USER_DEFINED(4). UNUSED(5) pad tokens are not matched. */
  size_t ns = 0;
  for (size_t id = 0; id < p.ntokens; id++)
    if (p.token_types[id] != 1 && p.token_types[id] != 6) {
      tz->special[id] = 1;
      ns++;
    }
  tz->num_special = ns;
  tz->special_ids = malloc(ns * sizeof(uint32_t));
  for (size_t id = 0, s = 0; id < p.ntokens; id++)
    if (tz->special[id]) tz->special_ids[s++] = (uint32_t)id;

  /* String->id map over keepable tokens (byte tokens + merges). */
  size_t keepable = p.ntokens - ns;
  size_t cap = 1;
  while (cap < keepable * 2) cap <<= 1;
  tz->map_cap = cap;
  tz->map = calloc(cap, sizeof(Entry));
  for (size_t id = 0; id < p.ntokens; id++)
    if (!tz->special[id]) smap_put(tz->map, cap, tz->tokens[id], id);

  /* byte <-> unicode tables (GPT-2 bytes_to_unicode). */
  uint8_t kept[256] = {0};
  for (int b = '!'; b <= '~'; b++) kept[b] = 1;
  for (int b = 0xA1; b <= 0xAC; b++) kept[b] = 1;
  for (int b = 0xAE; b <= 0xFF; b++) kept[b] = 1;
  for (int b = 0, n = 0; b < 256; b++) tz->byte_cp[b] = kept[b] ? (uint32_t)b : 256u + n++;
  for (int b = 0; b < 256; b++) tz->uni_to_byte[tz->byte_cp[b]] = (uint32_t)b;
  for (size_t id = 0; id < 256 && id < p.ntokens; id++) {
    size_t cl;
    uint32_t cp = utf8_cp(tz->tokens[id], &cl);
    tz->byte_to_id[tz->uni_to_byte[cp]] = (uint16_t)id;
  }

  free(p.token_types);
  return tz;
}

void qwen_tokenizer_free(QwenTokenizer *tz) {
  if (!tz) return;
  for (size_t i = 0; i < tz->vocab_size; i++) free(tz->tokens[i]);
  free(tz->tokens);
  free(tz->special);
  free(tz->special_ids);
  free(tz->map);
  free(tz);
}

size_t qwen_tokenize(const QwenTokenizer *tz, const char *text, uint32_t **out_ids) {
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
    uint32_t *tmp = malloc((out.n + 1) * sizeof(uint32_t));
    tmp[0] = tz->bos_token_id;
    memcpy(tmp + 1, out.v, out.n * sizeof(uint32_t));
    free(out.v);
    out.v = tmp;
    out.n++;
  }
  return out.n;
}

char *qwen_detokenize(const QwenTokenizer *tz, const uint32_t *ids, size_t n) {
  size_t total = 0;
  for (size_t k = 0; k < n; k++) {
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
  char *out = malloc(total + 1);
  size_t o = 0;
  for (size_t k = 0; k < n; k++) {
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
