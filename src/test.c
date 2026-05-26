#include "test.h"
#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Returns the current time in seconds with high resolution, using
// CLOCK_MONOTONIC
static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

// Fills a matrix with deterministic values based on row and column indices, for
// testing
static void fill_matrix(struct Matrix *m) {
  for (size_t i = 0; i < m->rows; ++i) {
    for (size_t j = 0; j < m->cols; ++j) {
      m->data[i * m->cols + j] = (int)((i + j) % 10 + 1);
    }
  }
}

// Comparison function for qsort to sort an array of doubles, used for median
// calculation
static int compare_double(const void *a, const void *b) {
  double x = *(const double *)a;
  double y = *(const double *)b;

  if (x < y) {
    return -1;
  } else if (x > y) {
    return 1;
  } else {
    return 0;
  }
}

// Computes the median of an array of doubles, used to summarize benchmark
// results
static double median(double *values, size_t n) {
  qsort(values, n, sizeof(double), compare_double);

  if (n % 2 == 1) {
    return values[n / 2];
  }
  return (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

// Runs benchmarks for naive and threaded matrix multiplication, measuring
// execution
void run_benchmark(size_t rows, size_t inner, size_t cols, size_t num_threads,
                   size_t iterations) {
  printf("\n[benchmark]\n");
  printf("A: %zux%zu, B: %zux%zu, threads: %zu, iterations: %zu\n", rows, inner,
         inner, cols, num_threads, iterations);

  if (iterations == 0) {
    fprintf(stderr, "iterations must be greater than zero\n");
    return;
  }

  struct Matrix a = init_matrix(rows, inner);
  struct Matrix b = init_matrix(inner, cols);

  if (!a.data || !b.data) {
    fprintf(stderr, "benchmark matrix allocation failed\n");
    free_matrix(&a);
    free_matrix(&b);
    return;
  }

  fill_matrix(&a);
  fill_matrix(&b);

  double *naive_times = malloc(sizeof(double) * iterations);
  double *threaded_times = malloc(sizeof(double) * iterations);

  if (!naive_times || !threaded_times) {
    fprintf(stderr, "memory allocation failed\n");
    free(naive_times);
    free(threaded_times);
    free_matrix(&a);
    free_matrix(&b);
    return;
  }

  for (size_t i = 0; i < iterations; ++i) {
    double start = now_seconds();
    struct Matrix c = matmul_naive(&a, &b);
    double end = now_seconds();

    if (!c.data) {
      fprintf(stderr, "naive benchmark failed\n");
      free_matrix(&c);
      goto cleanup;
    }

    naive_times[i] = end - start;
    free_matrix(&c);
  }

  for (size_t i = 0; i < iterations; ++i) {
    double start = now_seconds();
    struct Matrix c = matmul_threaded(&a, &b, num_threads);
    double end = now_seconds();

    if (!c.data) {
      fprintf(stderr, "threaded benchmark failed\n");
      free_matrix(&c);
      goto cleanup;
    }

    threaded_times[i] = end - start;
    free_matrix(&c);
  }

  double naive_median = median(naive_times, iterations);
  double threaded_median = median(threaded_times, iterations);

  printf("naive median:    %.6f sec\n", naive_median);
  printf("threaded median: %.6f sec\n", threaded_median);

  if (threaded_median > 0.0) {
    printf("speedup:         %.2fx\n", naive_median / threaded_median);
  }

cleanup:
  free(naive_times);
  free(threaded_times);
  free_matrix(&a);
  free_matrix(&b);
}
