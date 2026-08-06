#ifndef TK_BACKEND_H
#define TK_BACKEND_H

#include "tk_common.h"
#include "tk_matrix.h"

#include <stdbool.h>
#include <stddef.h>

typedef enum {
  TK_BACKEND_SINGLE,
  TK_BACKEND_PTHREAD,
  TK_BACKEND_OPENMP,
} tk_backend;

typedef enum {
  TK_LOOP_IJK,
  TK_LOOP_IKJ,
} tk_loop;

typedef struct {
  tk_backend backend;
  tk_loop loop_order;
  bool use_blocking;
  bool use_simd;
  size_t num_threads;
  size_t block_size;
} tk_matmul_cfg;

tk_matmul_cfg tk_matmul_cfg_new(tk_backend backend, tk_loop loop_order, bool use_blocking, bool use_simd, size_t num_threads, size_t block_size);

/* Fail-capable: returns/feeds *status. On failure TK_LOGE has already been
 * emitted by the failing layer (validation or backend). */
tk_status tk_mat_mul_into(const TkMatrix *lhs, const TkMatrix *rhs, TkMatrix *out, const tk_matmul_cfg *cfg, tk_status *status);
TkMatrix tk_mat_mul(const TkMatrix *lhs, const TkMatrix *rhs, const tk_matmul_cfg *cfg, tk_status *status);

const char *tk_backend_name(tk_backend backend);
const char *tk_loop_name(tk_loop loop_order);
bool tk_sse_available(void);
int tk_matmul_cfg_label(const tk_matmul_cfg *cfg, char *label, size_t label_size);

/* Validation: return TK_OK when valid, else TK_ERR_ARG and TK_LOGE once. */
tk_status tk_mat_validate(const TkMatrix *lhs, const TkMatrix *rhs, const TkMatrix *out);
tk_status tk_mat_input_valid(const TkMatrix *lhs, const TkMatrix *rhs);
tk_status tk_matmul_cfg_valid(const tk_matmul_cfg *cfg);

#endif /* TK_BACKEND_H */
