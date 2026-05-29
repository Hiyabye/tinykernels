#include "bench.h"
#include "matmul.h"
#include "matrix.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LABEL_SIZE 64

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int compare_double(const void *a, const void *b) {
  double x = *(const double *)a;
  double y = *(const double *)b;

  if (x < y) {
    return -1;
  }
  if (x > y) {
    return 1;
  }
  return 0;
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

static double bench_config(size_t rows, size_t inner, size_t cols, MatmulConfig cfg, size_t iterations) {
  Matrix a = matrix_new(rows, inner);
  Matrix b = matrix_new(inner, cols);
  Matrix c = matrix_new(rows, cols);

  if (!a.data || !b.data || !c.data) {
    matrix_free(&a);
    matrix_free(&b);
    matrix_free(&c);
    return -1.0;
  }

  matrix_fill_pattern(&a);
  matrix_fill_pattern(&b);

  double *times = malloc(sizeof(*times) * iterations);
  if (!times) {
    matrix_free(&a);
    matrix_free(&b);
    matrix_free(&c);
    return -1.0;
  }

  double result = -1.0;
  for (size_t i = 0; i < iterations; ++i) {
    double start = now_seconds();
    int ok = matmul_into(&a, &b, &c, cfg);
    double end = now_seconds();

    if (!ok) {
      goto cleanup;
    }

    times[i] = end - start;
  }

  result = median(times, iterations);

cleanup:
  free(times);
  matrix_free(&a);
  matrix_free(&b);
  matrix_free(&c);
  return result;
}

static void write_result(FILE *output, const char *sweep, size_t rows, size_t inner, size_t cols, MatmulConfig cfg,
                         size_t iterations, double time_sec, double baseline_sec) {
  char label[LABEL_SIZE];
  if (!matmul_config_label(cfg, label, sizeof(label))) {
    snprintf(label, sizeof(label), "unknown");
  }

  double speedup = time_sec > 0.0 ? baseline_sec / time_sec : 0.0;

  fprintf(output, "%s,%zu,%zu,%zu,%s,%s,%d,%zu,%zu,%zu,%f,%f,%s\n", sweep, rows, inner, cols,
          matmul_backend_name(cfg.backend), matmul_loop_order_name(cfg.loop_order), cfg.use_blocking, cfg.num_threads,
          cfg.block_size, iterations, time_sec, speedup, label);
}

static void bench_run_case(const char *sweep, size_t rows, size_t inner, size_t cols, size_t iterations,
                           const MatmulConfig *configs, size_t config_count, FILE *output) {
  if (iterations == 0 || !configs || config_count == 0 || !output) {
    fprintf(stderr, "invalid benchmark case\n");
    return;
  }

  printf("\n[benchmark] %s: A=%zux%zu, B=%zux%zu, iterations=%zu\n", sweep, rows, inner, inner, cols, iterations);
  printf("-----------------------------------------------------\n");

  double baseline = bench_config(rows, inner, cols, configs[0], iterations);
  if (baseline < 0.0) {
    fprintf(stderr, "benchmark failed\n");
    return;
  }

  for (size_t i = 0; i < config_count; ++i) {
    char label[LABEL_SIZE];
    if (!matmul_config_label(configs[i], label, sizeof(label))) {
      snprintf(label, sizeof(label), "unknown");
    }

    double time_sec = i == 0 ? baseline : bench_config(rows, inner, cols, configs[i], iterations);
    if (time_sec < 0.0) {
      fprintf(stderr, "benchmark failed for %s\n", label);
      return;
    }

    double speedup = time_sec > 0.0 ? baseline / time_sec : 0.0;
    printf("%-30s: %.6f sec (%.2fx)\n", label, time_sec, speedup);
    write_result(output, sweep, rows, inner, cols, configs[i], iterations, time_sec, baseline);
  }

  printf("-----------------------------------------------------\n");
}

static void bench_matrix_size_sweep(FILE *output) {
  const size_t ns[] = {64, 128, 256, 512, 1024};
  const size_t iterations = 10;
  const size_t threads = 4;
  const size_t block = 32;

  for (size_t idx = 0; idx < sizeof(ns) / sizeof(ns[0]); ++idx) {
    size_t n = ns[idx];
    MatmulConfig configs[] = {
        matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IJK, 0, 1, 1),
        matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IJK, 0, threads, 1),
        matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IJK, 1, 1, block),
        matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IJK, 1, threads, block),
#if TK_ENABLE_OPENMP
        matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IJK, 0, threads, 1),
        matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IJK, 1, threads, block),
#endif
    };

    bench_run_case("matrix_size", n, n, n, iterations, configs, sizeof(configs) / sizeof(configs[0]), output);
  }
}

static void bench_thread_count_sweep(FILE *output) {
  const size_t n = 512;
  const size_t iterations = 10;
  const size_t thread_counts[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const size_t block = 32;

  for (size_t idx = 0; idx < sizeof(thread_counts) / sizeof(thread_counts[0]); ++idx) {
    size_t threads = thread_counts[idx];
    MatmulConfig configs[] = {
        matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IJK, 0, threads, 1),
        matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IJK, 1, threads, block),
#if TK_ENABLE_OPENMP
        matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IJK, 0, threads, 1),
        matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IJK, 1, threads, block),
#endif
    };

    bench_run_case("thread_count", n, n, n, iterations, configs, sizeof(configs) / sizeof(configs[0]), output);
  }
}

static void bench_block_size_sweep(FILE *output) {
  const size_t n = 512;
  const size_t iterations = 10;
  const size_t threads = 4;
  const size_t block_sizes[] = {2, 4, 8, 16, 32, 64, 128};

  for (size_t idx = 0; idx < sizeof(block_sizes) / sizeof(block_sizes[0]); ++idx) {
    size_t block = block_sizes[idx];
    MatmulConfig configs[] = {
        matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IJK, 1, 1, block),
        matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IJK, 1, threads, block),
        matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IKJ, 1, 1, block),
        matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IKJ, 1, threads, block),
#if TK_ENABLE_OPENMP
        matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IJK, 1, threads, block),
        matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IKJ, 1, threads, block),
#endif
    };

    bench_run_case("block_size", n, n, n, iterations, configs, sizeof(configs) / sizeof(configs[0]), output);
  }
}

void bench_run_default_suite(const char *output_csv) {
  const char *path = output_csv ? output_csv : "benchmark_results.csv";
  FILE *output = fopen(path, "w");
  if (!output) {
    fprintf(stderr, "failed to open %s for writing\n", path);
    return;
  }

  fprintf(output, "sweep,rows,inner,cols,backend,loop_order,use_blocking,num_threads,block_size,iterations,time_sec,"
                  "speedup_vs_baseline,label\n");

  bench_matrix_size_sweep(output);
  bench_thread_count_sweep(output);
  bench_block_size_sweep(output);

  fclose(output);
  printf("\nWrote benchmark results to %s\n", path);
}
