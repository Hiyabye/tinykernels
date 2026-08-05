#include "bench_matmul.h"
#include "matmul.h"
#include "matrix.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define LABEL_SIZE 64
#define CONFIG_CAPACITY 24

static void add_config(MatmulConfig *configs, size_t *count, MatmulConfig config) {
  if (*count < CONFIG_CAPACITY) {
    configs[*count] = config;
    *count += 1;
  }
}

static double now_seconds(void) {
  struct timeval now;
  gettimeofday(&now, NULL);
  return now.tv_sec + now.tv_usec / 1e6;
}

static int compare_double(const void *lhs_ptr, const void *rhs_ptr) {
  double lhs = *(const double *)lhs_ptr;
  double rhs = *(const double *)rhs_ptr;

  if (lhs < rhs) return -1;
  if (lhs > rhs) return 1;
  return 0;
}

static double median(double *values, size_t value_count) {
  if (!values || value_count == 0) return -1.0;

  qsort(values, value_count, sizeof(double), compare_double);

  size_t mid = value_count / 2;
  if (value_count % 2 == 1) return values[mid];
  return (values[mid - 1] + values[mid]) / 2.0;
}

static double bench_config(size_t rows, size_t inner, size_t cols, MatmulConfig config, size_t iterations) {
  Matrix lhs = matrix_new(rows, inner);
  Matrix rhs = matrix_new(inner, cols);
  Matrix out = matrix_new(rows, cols);

  if (!lhs.data || !rhs.data || !out.data) {
    matrix_free(&lhs);
    matrix_free(&rhs);
    matrix_free(&out);
    return -1.0;
  }

  matrix_fill_pattern(&lhs);
  matrix_fill_pattern(&rhs);

  double *times = malloc(sizeof(*times) * iterations);
  if (!times) {
    matrix_free(&lhs);
    matrix_free(&rhs);
    matrix_free(&out);
    return -1.0;
  }

  double result = -1.0;
  for (size_t iter = 0; iter < iterations; ++iter) {
    double start_sec = now_seconds();
    int multiplied = matmul_into(&lhs, &rhs, &out, config);
    double end_sec = now_seconds();

    if (!multiplied) goto cleanup;

    times[iter] = end_sec - start_sec;
  }

  result = median(times, iterations);

cleanup:
  free(times);
  matrix_free(&lhs);
  matrix_free(&rhs);
  matrix_free(&out);
  return result;
}

static void write_result(FILE *output, const char *sweep, size_t rows, size_t inner, size_t cols, MatmulConfig config,
                         size_t iterations, double time_sec, double baseline_sec) {
  char label[LABEL_SIZE];
  if (!matmul_config_label(config, label, sizeof(label))) snprintf(label, sizeof(label), "unknown");

  double speedup = time_sec > 0.0 ? baseline_sec / time_sec : 0.0;

  fprintf(output, "%s,%zu,%zu,%zu,%s,%s,%d,%d,%zu,%zu,%zu,%f,%f,%s\n", sweep, rows, inner, cols,
          matmul_backend_name(config.backend), matmul_loop_order_name(config.loop_order), config.use_blocking,
          config.use_simd, config.num_threads, config.block_size, iterations, time_sec, speedup, label);
}

static void bench_run_case(const char *sweep, size_t rows, size_t inner, size_t cols, size_t iterations,
                           const MatmulConfig *configs, size_t config_count, FILE *output) {
  if (iterations == 0 || !configs || config_count == 0 || !output) {
    fprintf(stderr, "invalid benchmark case\n");
    return;
  }

  printf("\n[benchmark] %s: A=%zux%zu, B=%zux%zu, iterations=%zu\n", sweep, rows, inner, inner, cols, iterations);
  printf("-----------------------------------------------------\n");

  double baseline_sec = bench_config(rows, inner, cols, configs[0], iterations);
  if (baseline_sec < 0.0) {
    fprintf(stderr, "benchmark failed\n");
    return;
  }

  for (size_t config_idx = 0; config_idx < config_count; ++config_idx) {
    char label[LABEL_SIZE];
    if (!matmul_config_label(configs[config_idx], label, sizeof(label))) snprintf(label, sizeof(label), "unknown");

    double time_sec = config_idx == 0 ? baseline_sec : bench_config(rows, inner, cols, configs[config_idx], iterations);
    if (time_sec < 0.0) {
      fprintf(stderr, "benchmark failed for %s\n", label);
      return;
    }

    double speedup = time_sec > 0.0 ? baseline_sec / time_sec : 0.0;
    printf("%-30s: %.6f sec (%.2fx)\n", label, time_sec, speedup);
    write_result(output, sweep, rows, inner, cols, configs[config_idx], iterations, time_sec, baseline_sec);
  }

  printf("-----------------------------------------------------\n");
}

static void bench_matrix_size_sweep(FILE *output) {
  const size_t matrix_sizes[] = {64, 128, 256, 512, 1024};
  const size_t iterations = 10;
  const size_t thread_count = 4;
  const size_t block_size = 32;

  for (size_t size_idx = 0; size_idx < sizeof(matrix_sizes) / sizeof(matrix_sizes[0]); ++size_idx) {
    size_t matrix_size = matrix_sizes[size_idx];
    MatmulConfig configs[CONFIG_CAPACITY];
    size_t config_count = 0;

    add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IJK, 0, 0, 1, 1));
    add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IJK, 0, 0, thread_count, 1));
    add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IJK, 1, 0, 1, block_size));
    add_config(configs, &config_count,
               matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IJK, 1, 0, thread_count, block_size));

    if (matmul_simd_available()) {
      add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IKJ, 0, 1, 1, 1));
      add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IKJ, 1, 1, 1, block_size));
      add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IKJ, 0, 1, thread_count, 1));
      add_config(configs, &config_count,
                 matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IKJ, 1, 1, thread_count, block_size));
    }

#if TK_ENABLE_OPENMP
    add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IJK, 0, 0, thread_count, 1));
    add_config(configs, &config_count,
               matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IJK, 1, 0, thread_count, block_size));

    if (matmul_simd_available()) {
      add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IKJ, 0, 1, thread_count, 1));
      add_config(configs, &config_count,
                 matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IKJ, 1, 1, thread_count, block_size));
    }
#endif

    bench_run_case("matrix_size", matrix_size, matrix_size, matrix_size, iterations, configs, config_count, output);
  }
}

static void bench_thread_count_sweep(FILE *output) {
  const size_t matrix_size = 512;
  const size_t iterations = 10;
  const size_t thread_counts[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const size_t block_size = 32;

  for (size_t thread_idx = 0; thread_idx < sizeof(thread_counts) / sizeof(thread_counts[0]); ++thread_idx) {
    size_t thread_count = thread_counts[thread_idx];
    MatmulConfig configs[CONFIG_CAPACITY];
    size_t config_count = 0;

    add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IJK, 0, 0, thread_count, 1));
    add_config(configs, &config_count,
               matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IJK, 1, 0, thread_count, block_size));

    if (matmul_simd_available()) {
      add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IKJ, 0, 1, thread_count, 1));
      add_config(configs, &config_count,
                 matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IKJ, 1, 1, thread_count, block_size));
    }

#if TK_ENABLE_OPENMP
    add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IJK, 0, 0, thread_count, 1));
    add_config(configs, &config_count,
               matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IJK, 1, 0, thread_count, block_size));

    if (matmul_simd_available()) {
      add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IKJ, 0, 1, thread_count, 1));
      add_config(configs, &config_count,
                 matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IKJ, 1, 1, thread_count, block_size));
    }
#endif

    bench_run_case("thread_count", matrix_size, matrix_size, matrix_size, iterations, configs, config_count, output);
  }
}

static void bench_block_size_sweep(FILE *output) {
  const size_t matrix_size = 512;
  const size_t iterations = 10;
  const size_t thread_count = 4;
  const size_t block_sizes[] = {2, 4, 8, 16, 32, 64, 128};

  for (size_t block_idx = 0; block_idx < sizeof(block_sizes) / sizeof(block_sizes[0]); ++block_idx) {
    size_t block_size = block_sizes[block_idx];
    MatmulConfig configs[CONFIG_CAPACITY];
    size_t config_count = 0;

    add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IJK, 1, 0, 1, block_size));
    add_config(configs, &config_count,
               matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IJK, 1, 0, thread_count, block_size));
    add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IKJ, 1, 0, 1, block_size));
    add_config(configs, &config_count,
               matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IKJ, 1, 0, thread_count, block_size));

    if (matmul_simd_available()) {
      add_config(configs, &config_count, matmul_config(MATMUL_BACKEND_SINGLE, MATMUL_LOOP_IKJ, 1, 1, 1, block_size));
      add_config(configs, &config_count,
                 matmul_config(MATMUL_BACKEND_PTHREAD, MATMUL_LOOP_IKJ, 1, 1, thread_count, block_size));
    }

#if TK_ENABLE_OPENMP
    add_config(configs, &config_count,
               matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IJK, 1, 0, thread_count, block_size));
    add_config(configs, &config_count,
               matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IKJ, 1, 0, thread_count, block_size));

    if (matmul_simd_available()) {
      add_config(configs, &config_count,
                 matmul_config(MATMUL_BACKEND_OPENMP, MATMUL_LOOP_IKJ, 1, 1, thread_count, block_size));
    }
#endif

    bench_run_case("block_size", matrix_size, matrix_size, matrix_size, iterations, configs, config_count, output);
  }
}

void bench_run_default_suite(const char *output_csv) {
  const char *path = output_csv ? output_csv : "results/data/benchmark_results.csv";
  FILE *output = fopen(path, "w");
  if (!output) {
    fprintf(stderr, "failed to open %s for writing\n", path);
    return;
  }

  fprintf(output,
          "sweep,rows,inner,cols,backend,loop_order,use_blocking,use_simd,num_threads,block_size,iterations,time_sec,"
          "speedup_vs_baseline,label\n");

  bench_matrix_size_sweep(output);
  bench_thread_count_sweep(output);
  bench_block_size_sweep(output);

  fclose(output);
  printf("\nWrote benchmark results to %s\n", path);
}
