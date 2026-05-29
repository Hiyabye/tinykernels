#include "kernels.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

struct PthreadArgs {
  const Matrix *a;
  const Matrix *b;
  Matrix *c;
  MatmulConfig cfg;
  size_t row_start;
  size_t row_end;
};

static void *pthread_worker(void *arg) {
  struct PthreadArgs *args = arg;
  MatmulConfig cfg = args->cfg;

  if (cfg.use_blocking) {
    if (cfg.loop_order == MATMUL_LOOP_IJK) {
      tk_matmul_range_blocked_ijk(args->a, args->b, args->c, args->row_start, args->row_end, cfg.block_size);
    } else {
      tk_matmul_range_blocked_ikj(args->a, args->b, args->c, args->row_start, args->row_end, cfg.block_size);
    }
    return NULL;
  }

  if (cfg.loop_order == MATMUL_LOOP_IJK) {
    tk_matmul_range_ijk(args->a, args->b, args->c, args->row_start, args->row_end);
  } else {
    tk_matmul_range_ikj(args->a, args->b, args->c, args->row_start, args->row_end);
  }

  return NULL;
}

int tk_matmul_pthread_into(const Matrix *a, const Matrix *b, Matrix *c, MatmulConfig cfg) {
  size_t num_threads = cfg.num_threads;
  if (num_threads > a->rows) {
    num_threads = a->rows;
  }

  pthread_t *threads = malloc(sizeof(*threads) * num_threads);
  struct PthreadArgs *args = malloc(sizeof(*args) * num_threads);
  if (!threads || !args) {
    fprintf(stderr, "memory allocation failed\n");
    free(threads);
    free(args);
    return 0;
  }

  size_t rows_per_thread = a->rows / num_threads;
  size_t remainder = a->rows % num_threads;
  size_t current_row = 0;

  for (size_t t = 0; t < num_threads; ++t) {
    size_t rows_for_thread = rows_per_thread + (t < remainder ? 1 : 0);

    args[t].a = a;
    args[t].b = b;
    args[t].c = c;
    args[t].cfg = cfg;
    args[t].row_start = current_row;
    args[t].row_end = current_row + rows_for_thread;
    current_row = args[t].row_end;

    int err = pthread_create(&threads[t], NULL, pthread_worker, &args[t]);
    if (err != 0) {
      fprintf(stderr, "pthread_create failed\n");
      for (size_t joined = 0; joined < t; ++joined) {
        pthread_join(threads[joined], NULL);
      }
      free(threads);
      free(args);
      return 0;
    }
  }

  int ok = 1;
  for (size_t t = 0; t < num_threads; ++t) {
    if (pthread_join(threads[t], NULL) != 0) {
      ok = 0;
    }
  }

  free(threads);
  free(args);
  return ok;
}
