#include "tk_backend.h"
#include "tk_kernels.h"

#include <stdio.h>

tk_matmul_cfg tk_matmul_cfg_new(tk_backend backend, tk_loop loop_order, bool use_blocking, bool use_simd, size_t num_threads, size_t block_size) {
  return (tk_matmul_cfg){
      .backend = backend,
      .loop_order = loop_order,
      .use_blocking = use_blocking,
      .use_simd = use_simd,
      .num_threads = num_threads,
      .block_size = block_size,
  };
}

tk_status tk_mat_mul_into(const TkMatrix *lhs, const TkMatrix *rhs, TkMatrix *out, const tk_matmul_cfg *cfg, tk_status *status) {
  tk_status st = tk_mat_validate(lhs, rhs, out);
  if (st == TK_OK) st = tk_matmul_cfg_valid(cfg);
  if (st != TK_OK) {
    if (status) *status = st;
    return st;
  }

  tk_mat_fill(out, 0.0f);

  switch (cfg->backend) {
  case TK_BACKEND_SINGLE:
    st = tk_backend_single(lhs, rhs, out, cfg);
    break;
  case TK_BACKEND_PTHREAD:
    st = tk_backend_pthread(lhs, rhs, out, cfg);
    break;
  case TK_BACKEND_OPENMP:
    st = tk_backend_openmp(lhs, rhs, out, cfg);
    break;
  default:
    TK_LOGE("invalid matmul backend");
    st = TK_ERR_ARG;
    break;
  }

  if (status) *status = st;
  return st;
}

TkMatrix tk_mat_mul(const TkMatrix *lhs, const TkMatrix *rhs, const tk_matmul_cfg *cfg, tk_status *status) {
  if (status) *status = TK_OK;

  tk_status st = tk_mat_input_valid(lhs, rhs);
  if (st == TK_OK) st = tk_matmul_cfg_valid(cfg);
  if (st != TK_OK) {
    if (status) *status = st;
    return (TkMatrix){0, 0, NULL};
  }

  TkMatrix out = tk_mat_new(lhs->rows, rhs->cols);
  st = tk_mat_mul_into(lhs, rhs, &out, cfg, status);
  if (st != TK_OK) {
    tk_mat_free(&out);
    return (TkMatrix){0, 0, NULL};
  }

  return out;
}

const char *tk_backend_name(tk_backend backend) {
  switch (backend) {
  case TK_BACKEND_SINGLE:
    return "single";
  case TK_BACKEND_PTHREAD:
    return "pthread";
  case TK_BACKEND_OPENMP:
    return "openmp";
  default:
    return "unknown";
  }
}

const char *tk_loop_name(tk_loop loop_order) {
  switch (loop_order) {
  case TK_LOOP_IJK:
    return "ijk";
  case TK_LOOP_IKJ:
    return "ikj";
  default:
    return "unknown";
  }
}

bool tk_sse_available(void) {
#if defined(__SSE__)
  return sizeof(tk_elem_t) == sizeof(float);
#else
  return false;
#endif
}

int tk_matmul_cfg_label(const tk_matmul_cfg *cfg, char *label, size_t label_size) {
  if (!label || label_size == 0) return 0;

  const char *backend_name = tk_backend_name(cfg->backend);
  const char *loop_order_name = tk_loop_name(cfg->loop_order);
  const char *blocking_name = cfg->use_blocking ? "blocked" : "plain";
  const char *simd_name = cfg->use_simd ? "simd" : "sisd";

  int written = snprintf(label, label_size, "%s_%s_%s_%s", backend_name, blocking_name, simd_name, loop_order_name);
  return written >= 0 && (size_t)written < label_size;
}
