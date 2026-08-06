#ifndef TK_COMMON_H
#define TK_COMMON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Unified error/status type for the whole codebase. TK_OK is the literal 0 so
 * that functions returning tk_status read naturally; any nonzero value is an
 * error. */

typedef enum {
  TK_OK = 0,
  TK_ERR_ARG,         /* null ptr, bad dimension, invalid config value */
  TK_ERR_IO,          /* fopen/fread failure */
  TK_ERR_FORMAT,      /* malformed GGUF / metadata / token id */
  TK_ERR_UNSUPPORTED, /* unknown quant type, feature compiled out */
  TK_ERR_INTERNAL,    /* pthread_create/join etc. */
} tk_status;

size_t tk_min(size_t a, size_t b);

/* Allocation helpers: abort (exit) on failure. Callers never branch on NULL. */
void *tk_xmalloc(size_t n);
void *tk_xcalloc(size_t n, size_t sz);
void *tk_xrealloc(void *p, size_t n);
char *tk_xstrdup(const char *s);

/* Diagnostics always go to stderr, never stdout. */
void tk_log_error(const char *file, int line, const char *fmt, ...);
void tk_log_info(const char *fmt, ...);
#define TK_LOGE(...) tk_log_error(__FILE__, __LINE__, __VA_ARGS__)

double tk_now_seconds(void);

#endif /* TK_COMMON_H */
