#include "bench.h"
#include "matmul.h"
#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void fill_matrix(Matrix *m) {
  for (size_t i = 0; i < m->rows; ++i) {
    for (size_t j = 0; j < m->cols; ++j) {
      m->data[i * m->cols + j] = (mat_elem_t)((i + j) % 10 + 1);
    }
  }
}

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

static double median(double *values, size_t n) {
  if (!values || n == 0) {
    return -1.0;
  }

  qsort(values, n, sizeof(double), compare_double);

  if (n % 2 == 1) {
    return values[n / 2];
  }
  return (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

static double bench_matmul_ref_ijk(size_t rows, size_t inner, size_t cols,
                                   size_t iterations) {
  Matrix a = init_matrix(rows, inner);
  Matrix b = init_matrix(inner, cols);

  if (!a.data || !b.data) {
    fprintf(stderr, "benchmark matrix allocation failed\n");
    free_matrix(&a);
    free_matrix(&b);
    return -1.0;
  }

  fill_matrix(&a);
  fill_matrix(&b);

  double *naive_times = malloc(sizeof(double) * iterations);
  if (!naive_times) {
    fprintf(stderr, "memory allocation failed\n");
    free_matrix(&a);
    free_matrix(&b);
    return -1.0;
  }

  MatmulConfig cfg = {
      .kernel = MATMUL_REF_IJK,
      .num_threads = 1,
      .block_size = 1,
  };

  double result = -1.0;
  for (size_t i = 0; i < iterations; ++i) {
    double start = now_seconds();
    Matrix c = matmul(&a, &b, cfg);
    double end = now_seconds();

    if (!c.data) {
      fprintf(stderr, "naive benchmark failed\n");
      free_matrix(&c);
      goto cleanup;
    }

    naive_times[i] = end - start;
    free_matrix(&c);
  }

  result = median(naive_times, iterations);

cleanup:
  free(naive_times);
  free_matrix(&a);
  free_matrix(&b);

  return result;
}

static double bench_matmul_par_rows_ijk(size_t rows, size_t inner, size_t cols,
                                        size_t num_threads, size_t iterations) {
  Matrix a = init_matrix(rows, inner);
  Matrix b = init_matrix(inner, cols);

  if (!a.data || !b.data) {
    fprintf(stderr, "benchmark matrix allocation failed\n");
    free_matrix(&a);
    free_matrix(&b);
    return -1.0;
  }

  fill_matrix(&a);
  fill_matrix(&b);

  double *threaded_times = malloc(sizeof(double) * iterations);
  if (!threaded_times) {
    fprintf(stderr, "memory allocation failed\n");
    free_matrix(&a);
    free_matrix(&b);
    return -1.0;
  }

  MatmulConfig cfg = {
      .kernel = MATMUL_PAR_ROWS_IJK,
      .num_threads = num_threads,
      .block_size = 1,
  };

  double result = -1.0;
  for (size_t i = 0; i < iterations; ++i) {
    double start = now_seconds();
    Matrix c = matmul(&a, &b, cfg);
    double end = now_seconds();

    if (!c.data) {
      fprintf(stderr, "threaded benchmark failed\n");
      free_matrix(&c);
      goto cleanup;
    }

    threaded_times[i] = end - start;
    free_matrix(&c);
  }

  result = median(threaded_times, iterations);

cleanup:
  free(threaded_times);
  free_matrix(&a);
  free_matrix(&b);

  return result;
}

static double bench_matmul_seq_ikj(size_t rows, size_t inner, size_t cols,
                                   size_t iterations) {
  Matrix a = init_matrix(rows, inner);
  Matrix b = init_matrix(inner, cols);

  if (!a.data || !b.data) {
    fprintf(stderr, "benchmark matrix allocation failed\n");
    free_matrix(&a);
    free_matrix(&b);
    return -1.0;
  }

  fill_matrix(&a);
  fill_matrix(&b);

  double *ikj_times = malloc(sizeof(double) * iterations);
  if (!ikj_times) {
    fprintf(stderr, "memory allocation failed\n");
    free_matrix(&a);
    free_matrix(&b);
    return -1.0;
  }

  MatmulConfig cfg = {
      .kernel = MATMUL_SEQ_IKJ,
      .num_threads = 1,
      .block_size = 1,
  };

  double result = -1.0;
  for (size_t i = 0; i < iterations; ++i) {
    double start = now_seconds();
    Matrix c = matmul(&a, &b, cfg);
    double end = now_seconds();

    if (!c.data) {
      fprintf(stderr, "ikj benchmark failed\n");
      free_matrix(&c);
      goto cleanup;
    }

    ikj_times[i] = end - start;
    free_matrix(&c);
  }

  result = median(ikj_times, iterations);

cleanup:
  free(ikj_times);
  free_matrix(&a);
  free_matrix(&b);

  return result;
}

static double bench_matmul_seq_blocked_ikj(size_t rows, size_t inner,
                                           size_t cols, size_t block_size,
                                           size_t iterations) {
  if (block_size == 0) {
    fprintf(stderr, "block_size must be greater than zero\n");
    return -1.0;
  }

  Matrix a = init_matrix(rows, inner);
  Matrix b = init_matrix(inner, cols);

  if (!a.data || !b.data) {
    fprintf(stderr, "benchmark matrix allocation failed\n");
    free_matrix(&a);
    free_matrix(&b);
    return -1.0;
  }

  fill_matrix(&a);
  fill_matrix(&b);

  double *blocked_times = malloc(sizeof(double) * iterations);
  if (!blocked_times) {
    fprintf(stderr, "memory allocation failed\n");
    free_matrix(&a);
    free_matrix(&b);
    return -1.0;
  }

  MatmulConfig cfg = {
      .kernel = MATMUL_SEQ_BLOCKED_IKJ,
      .num_threads = 1,
      .block_size = block_size,
  };

  double result = -1.0;
  for (size_t i = 0; i < iterations; ++i) {
    double start = now_seconds();
    Matrix c = matmul(&a, &b, cfg);
    double end = now_seconds();

    if (!c.data) {
      fprintf(stderr, "blocked benchmark failed\n");
      free_matrix(&c);
      goto cleanup;
    }

    blocked_times[i] = end - start;
    free_matrix(&c);
  }

  result = median(blocked_times, iterations);

cleanup:
  free(blocked_times);
  free_matrix(&a);
  free_matrix(&b);

  return result;
}

static double bench_matmul_par_rows_blocked_ikj(size_t rows, size_t inner,
                                                size_t cols, size_t num_threads,
                                                size_t block_size,
                                                size_t iterations) {
  if (block_size == 0) {
    fprintf(stderr, "block_size must be greater than zero\n");
    return -1.0;
  }

  Matrix a = init_matrix(rows, inner);
  Matrix b = init_matrix(inner, cols);

  if (!a.data || !b.data) {
    fprintf(stderr, "benchmark matrix allocation failed\n");
    free_matrix(&a);
    free_matrix(&b);
    return -1.0;
  }

  fill_matrix(&a);
  fill_matrix(&b);

  double *threaded_blocked_times = malloc(sizeof(double) * iterations);
  if (!threaded_blocked_times) {
    fprintf(stderr, "memory allocation failed\n");
    free_matrix(&a);
    free_matrix(&b);
    return -1.0;
  }

  MatmulConfig cfg = {
      .kernel = MATMUL_PAR_ROWS_BLOCKED_IKJ,
      .num_threads = num_threads,
      .block_size = block_size,
  };

  double result = -1.0;
  for (size_t i = 0; i < iterations; ++i) {
    double start = now_seconds();
    Matrix c = matmul(&a, &b, cfg);
    double end = now_seconds();

    if (!c.data) {
      fprintf(stderr, "threaded blocked benchmark failed\n");
      free_matrix(&c);
      goto cleanup;
    }

    threaded_blocked_times[i] = end - start;
    free_matrix(&c);
  }
  result = median(threaded_blocked_times, iterations);

cleanup:
  free(threaded_blocked_times);
  free_matrix(&a);
  free_matrix(&b);

  return result;
}

void bench_run_case(size_t rows, size_t inner, size_t cols, size_t threads,
                    size_t block_size, size_t iterations) {
  printf("\n[benchmark]\n");
  printf("A: %zux%zu, B: %zux%zu, threads: %zu, block size: %zu, iterations: "
         "%zu\n",
         rows, inner, inner, cols, threads, block_size, iterations);

  if (iterations == 0) {
    fprintf(stderr, "iterations must be greater than zero\n");
    return;
  }

  double naive_median = bench_matmul_ref_ijk(rows, inner, cols, iterations);
  double threaded_median =
      bench_matmul_par_rows_ijk(rows, inner, cols, threads, iterations);
  double ikj_median = bench_matmul_seq_ikj(rows, inner, cols, iterations);
  double blocked_median =
      bench_matmul_seq_blocked_ikj(rows, inner, cols, block_size, iterations);
  double threaded_blocked_median = bench_matmul_par_rows_blocked_ikj(
      rows, inner, cols, threads, block_size, iterations);

  if (naive_median < 0.0 || threaded_median < 0.0 || ikj_median < 0.0 ||
      blocked_median < 0.0 || threaded_blocked_median < 0.0) {
    fprintf(stderr, "benchmark failed\n");
    return;
  }

  printf("naive median:             %.6f sec\n", naive_median);
  printf("threaded median:          %.6f sec\n", threaded_median);
  printf("ikj median:               %.6f sec\n", ikj_median);
  printf("blocked median:           %.6f sec\n", blocked_median);
  printf("threaded blocked median:  %.6f sec\n", threaded_blocked_median);

  if (threaded_median > 0.0) {
    printf("threaded speedup:         %.2fx\n", naive_median / threaded_median);
  }
  if (ikj_median > 0.0) {
    printf("ikj speedup:              %.2fx\n", naive_median / ikj_median);
  }
  if (blocked_median > 0.0) {
    printf("blocked speedup:          %.2fx\n", naive_median / blocked_median);
  }
  if (threaded_blocked_median > 0.0) {
    printf("threaded blocked speedup: %.2fx\n",
           naive_median / threaded_blocked_median);
  }
}

void bench_run_default_suite(void) {
  // matrix size sweep
  run_bench(128, 128, 128, 1, 128, 100);
  run_bench(256, 256, 256, 1, 256, 100);
  run_bench(512, 512, 512, 1, 512, 100);

  // thread count sweep
  run_bench(512, 512, 512, 2, 512, 100);
  run_bench(512, 512, 512, 4, 512, 100);
  run_bench(512, 512, 512, 8, 512, 100);

  // block size sweep
  run_bench(512, 512, 512, 1, 32, 100);
  run_bench(512, 512, 512, 1, 64, 100);
  run_bench(512, 512, 512, 1, 128, 100);
}
