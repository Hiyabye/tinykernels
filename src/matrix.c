#include "matrix.h"

#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Helper function to get the minimum of two size_t values, used for block size
static size_t min_size(size_t a, size_t b) { return a < b ? a : b; }

// Arguments structure for worker threads in matmul_threaded
struct ThreadArgs {
  const struct Matrix *a;
  const struct Matrix *b;
  struct Matrix *c;
  size_t row_start;
  size_t row_end;
  size_t block_size;
};

// Returns an empty matrix (0 rows, 0 cols, NULL data)
struct Matrix empty_matrix(void) {
  struct Matrix m = {0, 0, NULL};
  return m;
}

// Initializes a matrix with given rows and columns, allocates memory for data
struct Matrix init_matrix(size_t r, size_t c) {
  if (r == 0 || c == 0) {
    fprintf(stderr, "invalid matrix dimensions\n");
    return empty_matrix();
  }

  if (r > SIZE_MAX / c) {
    fprintf(stderr, "matrix size overflow\n");
    return empty_matrix();
  }

  struct Matrix m = {r, c, NULL};
  m.data = calloc(r * c, sizeof(int));
  if (!m.data) {
    fprintf(stderr, "memory allocation failed\n");
    return empty_matrix();
  }
  return m;
}

// Prints the contents of a matrix to stdout
void print_matrix(const struct Matrix *m) {
  for (size_t i = 0; i < m->rows; ++i) {
    for (size_t j = 0; j < m->cols; ++j) {
      printf("%d ", m->data[i * m->cols + j]);
    }
    printf("\n");
  }
}

// Frees the memory allocated for a matrix and resets its fields
void free_matrix(struct Matrix *m) {
  if (!m || !m->data) {
    return;
  }
  free(m->data);
  m->data = NULL;
  m->rows = 0;
  m->cols = 0;
}

// Naive matrix multiplication implementation (single-threaded)
struct Matrix matmul_naive(const struct Matrix *a, const struct Matrix *b) {
  if (!a || !b || !a->data || !b->data) {
    fprintf(stderr, "invalid matrix\n");
    return empty_matrix();
  }

  if (a->cols != b->rows) {
    fprintf(stderr, "incompatible matrix dimensions\n");
    return empty_matrix();
  }

  struct Matrix c = init_matrix(a->rows, b->cols);
  if (!c.data) {
    fprintf(stderr, "failed to initialize result matrix\n");
    return empty_matrix();
  }

  for (size_t i = 0; i < a->rows; ++i) {
    for (size_t j = 0; j < b->cols; ++j) {
      for (size_t k = 0; k < a->cols; ++k) {
        c.data[i * c.cols + j] +=
            a->data[i * a->cols + k] * b->data[k * b->cols + j];
      }
    }
  }
  return c;
}

// Worker function for each thread in matmul_threaded, computes a portion of the
// result matrix
static void *matmul_worker(void *arg) {
  struct ThreadArgs *args = arg;

  const struct Matrix *a = args->a;
  const struct Matrix *b = args->b;
  struct Matrix *c = args->c;

  for (size_t i = args->row_start; i < args->row_end; ++i) {
    for (size_t j = 0; j < b->cols; ++j) {
      int sum = 0;
      for (size_t k = 0; k < a->cols; ++k) {
        sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
      }
      c->data[i * c->cols + j] = sum;
    }
  }
  return NULL;
}

// Threaded matrix multiplication implementation, divides work among multiple
// threads
struct Matrix matmul_threaded(const struct Matrix *a, const struct Matrix *b,
                              size_t num_threads) {
  if (!a || !b || !a->data || !b->data) {
    fprintf(stderr, "invalid matrix\n");
    return empty_matrix();
  }

  if (a->cols != b->rows) {
    fprintf(stderr, "incompatible matrix dimensions\n");
    return empty_matrix();
  }

  if (num_threads == 0) {
    fprintf(stderr, "num_threads must be greater than zero\n");
    return empty_matrix();
  }

  if (num_threads > a->rows) {
    num_threads = a->rows;
  }

  struct Matrix c = init_matrix(a->rows, b->cols);
  if (!c.data) {
    fprintf(stderr, "failed to initialize result matrix\n");
    return empty_matrix();
  }

  pthread_t *threads = malloc(sizeof(pthread_t) * num_threads);
  struct ThreadArgs *args = malloc(sizeof(struct ThreadArgs) * num_threads);

  if (!threads || !args) {
    fprintf(stderr, "memory allocation failed\n");
    free(threads);
    free(args);
    free_matrix(&c);
    return empty_matrix();
  }

  size_t rows_per_thread = a->rows / num_threads;
  size_t remainder = a->rows % num_threads;

  size_t current_row = 0;

  for (size_t t = 0; t < num_threads; ++t) {
    size_t rows_for_this_thread = rows_per_thread + (t < remainder ? 1 : 0);

    args[t].a = a;
    args[t].b = b;
    args[t].c = &c;
    args[t].row_start = current_row;
    args[t].row_end = current_row + rows_for_this_thread;

    current_row = args[t].row_end;

    int err = pthread_create(&threads[t], NULL, matmul_worker, &args[t]);
    if (err != 0) {
      fprintf(stderr, "pthread_create failed\n");

      for (size_t joined = 0; joined < t; ++joined) {
        pthread_join(threads[joined], NULL);
      }

      free(threads);
      free(args);
      free_matrix(&c);
      return empty_matrix();
    }
  }

  for (size_t t = 0; t < num_threads; ++t) {
    pthread_join(threads[t], NULL);
  }

  free(threads);
  free(args);

  return c;
}

struct Matrix matmul_ikj(const struct Matrix *a, const struct Matrix *b) {
  if (!a || !b || !a->data || !b->data) {
    fprintf(stderr, "invalid matrix\n");
    return empty_matrix();
  }

  if (a->cols != b->rows) {
    fprintf(stderr, "incompatible matrix dimensions\n");
    return empty_matrix();
  }

  struct Matrix c = init_matrix(a->rows, b->cols);
  if (!c.data) {
    fprintf(stderr, "failed to initialize result matrix\n");
    return empty_matrix();
  }

  for (size_t i = 0; i < a->rows; ++i) {
    for (size_t k = 0; k < a->cols; ++k) {
      int aik = a->data[i * a->cols + k];
      for (size_t j = 0; j < b->cols; ++j) {
        c.data[i * c.cols + j] += aik * b->data[k * b->cols + j];
      }
    }
  }
  return c;
}

struct Matrix matmul_blocked(const struct Matrix *a, const struct Matrix *b,
                             size_t block_size) {
  if (!a || !b || !a->data || !b->data) {
    fprintf(stderr, "invalid matrix\n");
    return empty_matrix();
  }

  if (a->cols != b->rows) {
    fprintf(stderr, "incompatible matrix dimensions\n");
    return empty_matrix();
  }

  if (block_size == 0) {
    fprintf(stderr, "block_size must be greater than zero\n");
    return empty_matrix();
  }

  struct Matrix c = init_matrix(a->rows, b->cols);
  if (!c.data) {
    fprintf(stderr, "failed to initialize result matrix\n");
    return empty_matrix();
  }

  for (size_t i0 = 0; i0 < a->rows; i0 += block_size) {
    for (size_t k0 = 0; k0 < a->cols; k0 += block_size) {
      for (size_t j0 = 0; j0 < b->cols; j0 += block_size) {
        size_t i_max = min_size(i0 + block_size, a->rows);
        size_t k_max = min_size(k0 + block_size, a->cols);
        size_t j_max = min_size(j0 + block_size, b->cols);

        for (size_t i = i0; i < i_max; ++i) {
          for (size_t k = k0; k < k_max; ++k) {
            int aik = a->data[i * a->cols + k];

            for (size_t j = j0; j < j_max; ++j) {
              c.data[i * c.cols + j] += aik * b->data[k * b->cols + j];
            }
          }
        }
      }
    }
  }
  return c;
}

static void *matmul_blocked_worker(void *arg) {
  struct ThreadArgs *args = arg;

  const struct Matrix *a = args->a;
  const struct Matrix *b = args->b;
  struct Matrix *c = args->c;
  size_t block_size = args->block_size;

  for (size_t i0 = args->row_start; i0 < args->row_end; i0 += block_size) {
    for (size_t k0 = 0; k0 < a->cols; k0 += block_size) {
      for (size_t j0 = 0; j0 < b->cols; j0 += block_size) {
        size_t i_max = min_size(i0 + block_size, args->row_end);
        size_t k_max = min_size(k0 + block_size, a->cols);
        size_t j_max = min_size(j0 + block_size, b->cols);

        for (size_t i = i0; i < i_max; ++i) {
          for (size_t k = k0; k < k_max; ++k) {
            int aik = a->data[i * a->cols + k];

            for (size_t j = j0; j < j_max; ++j) {
              c->data[i * c->cols + j] += aik * b->data[k * b->cols + j];
            }
          }
        }
      }
    }
  }

  return NULL;
}

struct Matrix matmul_threaded_blocked(const struct Matrix *a,
                                      const struct Matrix *b,
                                      size_t num_threads, size_t block_size) {
  if (!a || !b || !a->data || !b->data) {
    fprintf(stderr, "invalid matrix\n");
    return empty_matrix();
  }

  if (a->cols != b->rows) {
    fprintf(stderr, "incompatible matrix dimensions\n");
    return empty_matrix();
  }

  if (num_threads == 0) {
    fprintf(stderr, "num_threads must be greater than zero\n");
    return empty_matrix();
  }

  if (block_size == 0) {
    fprintf(stderr, "block_size must be greater than zero\n");
    return empty_matrix();
  }

  if (num_threads > a->rows) {
    num_threads = a->rows;
  }

  struct Matrix c = init_matrix(a->rows, b->cols);
  if (!c.data) {
    fprintf(stderr, "failed to initialize result matrix\n");
    return empty_matrix();
  }

  pthread_t *threads = malloc(sizeof(pthread_t) * num_threads);
  struct ThreadArgs *args = malloc(sizeof(struct ThreadArgs) * num_threads);

  if (!threads || !args) {
    fprintf(stderr, "memory allocation failed\n");
    free(threads);
    free(args);
    free_matrix(&c);
    return empty_matrix();
  }

  size_t rows_per_thread = a->rows / num_threads;
  size_t remainder = a->rows % num_threads;

  size_t current_row = 0;

  for (size_t t = 0; t < num_threads; ++t) {
    size_t rows_for_this_thread = rows_per_thread + (t < remainder ? 1 : 0);

    args[t].a = a;
    args[t].b = b;
    args[t].c = &c;
    args[t].row_start = current_row;
    args[t].row_end = current_row + rows_for_this_thread;
    args[t].block_size = block_size;

    current_row = args[t].row_end;

    int err =
        pthread_create(&threads[t], NULL, matmul_blocked_worker, &args[t]);
    if (err != 0) {
      fprintf(stderr, "pthread_create failed\n");

      for (size_t joined = 0; joined < t; ++joined) {
        pthread_join(threads[joined], NULL);
      }

      free(threads);
      free(args);
      free_matrix(&c);
      return empty_matrix();
    }
  }

  for (size_t t = 0; t < num_threads; ++t) {
    pthread_join(threads[t], NULL);
  }

  free(threads);
  free(args);

  return c;
}
