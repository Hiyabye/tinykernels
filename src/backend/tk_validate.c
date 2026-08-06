#include "tk_backend.h"

tk_status tk_mat_validate(const TkMatrix *lhs, const TkMatrix *rhs, const TkMatrix *out) {
  if (!lhs || !rhs || !out || !lhs->data || !rhs->data || !out->data) {
    TK_LOGE("invalid matrix");
    return TK_ERR_ARG;
  }

  if (lhs->rows == 0 || lhs->cols == 0 || rhs->rows == 0 || rhs->cols == 0 || out->rows == 0 || out->cols == 0) {
    TK_LOGE("matrix dimensions must be greater than zero");
    return TK_ERR_ARG;
  }

  if (lhs->cols != rhs->rows) {
    TK_LOGE("incompatible input matrix dimensions");
    return TK_ERR_ARG;
  }

  if (out->rows != lhs->rows || out->cols != rhs->cols) {
    TK_LOGE("invalid output matrix dimensions");
    return TK_ERR_ARG;
  }

  return TK_OK;
}

tk_status tk_mat_input_valid(const TkMatrix *lhs, const TkMatrix *rhs) {
  if (!lhs || !rhs || !lhs->data || !rhs->data) {
    TK_LOGE("invalid matrix multiplication request");
    return TK_ERR_ARG;
  }

  if (lhs->rows == 0 || lhs->cols == 0 || rhs->rows == 0 || rhs->cols == 0) {
    TK_LOGE("matrix dimensions must be greater than zero");
    return TK_ERR_ARG;
  }

  if (lhs->cols != rhs->rows) {
    TK_LOGE("incompatible input matrix dimensions");
    return TK_ERR_ARG;
  }

  return TK_OK;
}

tk_status tk_matmul_cfg_valid(const tk_matmul_cfg *cfg) {
  if (!cfg) {
    TK_LOGE("null matmul config");
    return TK_ERR_ARG;
  }

  if (cfg->backend != TK_BACKEND_SINGLE && cfg->backend != TK_BACKEND_PTHREAD && cfg->backend != TK_BACKEND_OPENMP) {
    TK_LOGE("invalid matmul backend");
    return TK_ERR_ARG;
  }

  if (cfg->loop_order != TK_LOOP_IJK && cfg->loop_order != TK_LOOP_IKJ) {
    TK_LOGE("invalid matmul loop order");
    return TK_ERR_ARG;
  }

  if (cfg->backend != TK_BACKEND_SINGLE && cfg->num_threads == 0) {
    TK_LOGE("num_threads must be greater than zero");
    return TK_ERR_ARG;
  }

  if (cfg->use_blocking && cfg->block_size == 0) {
    TK_LOGE("block_size must be greater than zero");
    return TK_ERR_ARG;
  }

  if (cfg->use_simd && !tk_sse_available()) {
    TK_LOGE("SIMD support is unavailable on this target");
    return TK_ERR_ARG;
  }

  if (cfg->use_simd && cfg->loop_order != TK_LOOP_IKJ) {
    TK_LOGE("SIMD currently requires IKJ loop order");
    return TK_ERR_ARG;
  }

  return TK_OK;
}
