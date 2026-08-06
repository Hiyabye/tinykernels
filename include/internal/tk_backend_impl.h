#ifndef TK_BACKEND_IMPL_H
#define TK_BACKEND_IMPL_H

#include "tk_backend.h"

tk_status tk_backend_single(const TkMatrix *lhs, const TkMatrix *rhs, TkMatrix *out, const tk_matmul_cfg *cfg);
tk_status tk_backend_pthread(const TkMatrix *lhs, const TkMatrix *rhs, TkMatrix *out, const tk_matmul_cfg *cfg);
tk_status tk_backend_openmp(const TkMatrix *lhs, const TkMatrix *rhs, TkMatrix *out, const tk_matmul_cfg *cfg);

#endif /* TK_BACKEND_IMPL_H */
