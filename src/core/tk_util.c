#include "tk_common.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

size_t tk_min(size_t a, size_t b) { return a < b ? a : b; }

void *tk_xmalloc(size_t n) {
  void *p = malloc(n ? n : 1);
  if (!p) {
    TK_LOGE("out of memory");
    exit(EXIT_FAILURE);
  }
  return p;
}

void *tk_xcalloc(size_t n, size_t sz) {
  void *p = calloc(n ? n : 1, sz ? sz : 1);
  if (!p) {
    TK_LOGE("out of memory");
    exit(EXIT_FAILURE);
  }
  return p;
}

void *tk_xrealloc(void *p, size_t n) {
  void *q = realloc(p, n ? n : 1);
  if (!q) {
    TK_LOGE("out of memory");
    exit(EXIT_FAILURE);
  }
  return q;
}

char *tk_xstrdup(const char *s) {
  size_t n = strlen(s) + 1;
  return memcpy(tk_xmalloc(n), s, n);
}

void tk_log_error(const char *file, int line, const char *fmt, ...) {
  va_list ap;
  fprintf(stderr, "[error] %s:%d: ", file, line);
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
}

void tk_log_info(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
}

double tk_now_seconds(void) {
  struct timeval now;
  gettimeofday(&now, NULL);
  return now.tv_sec + now.tv_usec / 1e6;
}
