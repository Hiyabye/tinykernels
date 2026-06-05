#include "kernels.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

struct PthreadArgs {
  const Matrix *lhs;
  const Matrix *rhs;
  Matrix *out;
  MatmulConfig config;
  size_t row_start;
  size_t row_end;
};

static void *pthread_worker(void *worker_arg) {
  struct PthreadArgs *worker_args = worker_arg;
  tk_matmul_range(worker_args->lhs, worker_args->rhs, worker_args->out, worker_args->config, worker_args->row_start,
                  worker_args->row_end);
  return NULL;
}

int tk_matmul_pthread_into(const Matrix *lhs, const Matrix *rhs, Matrix *out, MatmulConfig config) {
  size_t worker_count = config.num_threads;
  if (worker_count > lhs->rows) {
    worker_count = lhs->rows;
  }

  pthread_t *threads = malloc(sizeof(*threads) * worker_count);
  struct PthreadArgs *worker_args = malloc(sizeof(*worker_args) * worker_count);
  if (!threads || !worker_args) {
    fprintf(stderr, "memory allocation failed\n");
    free(threads);
    free(worker_args);
    return 0;
  }

  size_t rows_per_worker = lhs->rows / worker_count;
  size_t extra_rows = lhs->rows % worker_count;
  size_t current_row = 0;

  for (size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
    size_t rows_for_worker = rows_per_worker + (worker_index < extra_rows ? 1 : 0);

    worker_args[worker_index].lhs = lhs;
    worker_args[worker_index].rhs = rhs;
    worker_args[worker_index].out = out;
    worker_args[worker_index].config = config;
    worker_args[worker_index].row_start = current_row;
    worker_args[worker_index].row_end = current_row + rows_for_worker;
    current_row = worker_args[worker_index].row_end;

    int create_error = pthread_create(&threads[worker_index], NULL, pthread_worker, &worker_args[worker_index]);
    if (create_error != 0) {
      fprintf(stderr, "pthread_create failed\n");
      for (size_t joined_index = 0; joined_index < worker_index; ++joined_index) {
        pthread_join(threads[joined_index], NULL);
      }
      free(threads);
      free(worker_args);
      return 0;
    }
  }

  int all_threads_joined = 1;
  for (size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
    if (pthread_join(threads[worker_index], NULL) != 0) {
      all_threads_joined = 0;
    }
  }

  free(threads);
  free(worker_args);
  return all_threads_joined;
}
