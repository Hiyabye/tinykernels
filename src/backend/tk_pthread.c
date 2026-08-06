#include "tk_common.h"
#include "tk_kernels.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

struct PthreadArgs {
  const TkMatrix *lhs;
  const TkMatrix *rhs;
  TkMatrix *out;
  const tk_matmul_cfg *cfg;
  size_t row_start;
  size_t row_end;
};

static void *pthread_worker(void *worker_arg) {
  struct PthreadArgs *worker_args = worker_arg;
  tk_kernel_run(worker_args->lhs, worker_args->rhs, worker_args->out, worker_args->cfg, worker_args->row_start, worker_args->row_end);
  return NULL;
}

tk_status tk_backend_pthread(const TkMatrix *lhs, const TkMatrix *rhs, TkMatrix *out, const tk_matmul_cfg *cfg) {
  size_t worker_count = cfg->num_threads;
  if (worker_count > lhs->rows) worker_count = lhs->rows;

  pthread_t *threads = tk_xmalloc(sizeof(*threads) * worker_count);
  struct PthreadArgs *worker_args = tk_xmalloc(sizeof(*worker_args) * worker_count);

  size_t rows_per_worker = lhs->rows / worker_count;
  size_t extra_rows = lhs->rows % worker_count;
  size_t current_row = 0;

  for (size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
    size_t rows_for_worker = rows_per_worker + (worker_index < extra_rows ? 1 : 0);

    worker_args[worker_index].lhs = lhs;
    worker_args[worker_index].rhs = rhs;
    worker_args[worker_index].out = out;
    worker_args[worker_index].cfg = cfg;
    worker_args[worker_index].row_start = current_row;
    worker_args[worker_index].row_end = current_row + rows_for_worker;
    current_row = worker_args[worker_index].row_end;

    int create_error = pthread_create(&threads[worker_index], NULL, pthread_worker, &worker_args[worker_index]);
    if (create_error != 0) {
      TK_LOGE("pthread_create: %s", strerror(create_error));
      for (size_t joined_index = 0; joined_index < worker_index; ++joined_index) { pthread_join(threads[joined_index], NULL); }
      free(threads);
      free(worker_args);
      return TK_ERR_INTERNAL;
    }
  }

  int all_threads_joined = 1;
  for (size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
    int join_error = pthread_join(threads[worker_index], NULL);
    if (join_error != 0) {
      TK_LOGE("pthread_join: %s", strerror(join_error));
      all_threads_joined = 0;
    }
  }

  free(threads);
  free(worker_args);
  return all_threads_joined ? TK_OK : TK_ERR_INTERNAL;
}
