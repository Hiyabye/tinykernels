#include "tk_bench.h"
#include "tk_backend.h"
#include "tk_common.h"
#include "tk_matrix.h"

#include <errno.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LABEL_SIZE 64
#define CONFIG_CAPACITY 24

static void add_config(tk_matmul_cfg *configs, size_t *count, tk_matmul_cfg config) {
  if (*count < CONFIG_CAPACITY) {
    configs[*count] = config;
    *count += 1;
  }
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

static double bench_config(size_t rows, size_t inner, size_t cols, const tk_matmul_cfg *cfg, size_t iterations) {
  TkMatrix lhs = tk_mat_new(rows, inner);
  TkMatrix rhs = tk_mat_new(inner, cols);
  TkMatrix out = tk_mat_new(rows, cols);

  if (!lhs.data || !rhs.data || !out.data) {
    tk_mat_free(&lhs);
    tk_mat_free(&rhs);
    tk_mat_free(&out);
    return -1.0;
  }

  tk_mat_fill_pattern(&lhs);
  tk_mat_fill_pattern(&rhs);

  double *times = tk_xmalloc(sizeof(*times) * iterations);

  double result = -1.0;
  for (size_t iter = 0; iter < iterations; ++iter) {
    double start_sec = tk_now_seconds();
    tk_status st = tk_mat_mul_into(&lhs, &rhs, &out, cfg, NULL);
    double end_sec = tk_now_seconds();

    if (st != TK_OK) goto cleanup;

    times[iter] = end_sec - start_sec;
  }

  result = median(times, iterations);

cleanup:
  free(times);
  tk_mat_free(&lhs);
  tk_mat_free(&rhs);
  tk_mat_free(&out);
  return result;
}

static void write_result(FILE *output, const char *sweep, size_t rows, size_t inner, size_t cols, const tk_matmul_cfg *cfg, size_t iterations,
                         double time_sec, double baseline_sec) {
  char label[LABEL_SIZE];
  if (!tk_matmul_cfg_label(cfg, label, sizeof(label))) snprintf(label, sizeof(label), "unknown");

  double speedup = time_sec > 0.0 ? baseline_sec / time_sec : 0.0;

  fprintf(output, "%s,%zu,%zu,%zu,%s,%s,%d,%d,%zu,%zu,%zu,%f,%f,%s\n", sweep, rows, inner, cols, tk_backend_name(cfg->backend),
          tk_loop_name(cfg->loop_order), cfg->use_blocking, cfg->use_simd, cfg->num_threads, cfg->block_size, iterations, time_sec, speedup, label);
}

static void bench_run_case(FILE *output, const char *sweep, size_t rows, size_t inner, size_t cols, size_t iterations, const tk_matmul_cfg *configs,
                           size_t config_count) {
  if (iterations == 0 || !configs || config_count == 0 || !output) {
    TK_LOGE("invalid benchmark case");
    return;
  }

  fprintf(output, "\n[benchmark] %s: A=%zux%zu, B=%zux%zu, iterations=%zu\n", sweep, rows, inner, inner, cols, iterations);
  fprintf(output, "-----------------------------------------------------\n");

  double baseline_sec = bench_config(rows, inner, cols, &configs[0], iterations);
  if (baseline_sec < 0.0) {
    TK_LOGE("benchmark failed");
    return;
  }

  for (size_t config_idx = 0; config_idx < config_count; ++config_idx) {
    char label[LABEL_SIZE];
    if (!tk_matmul_cfg_label(&configs[config_idx], label, sizeof(label))) snprintf(label, sizeof(label), "unknown");

    double time_sec = config_idx == 0 ? baseline_sec : bench_config(rows, inner, cols, &configs[config_idx], iterations);
    if (time_sec < 0.0) {
      TK_LOGE("benchmark failed for %s", label);
      return;
    }

    double speedup = time_sec > 0.0 ? baseline_sec / time_sec : 0.0;
    fprintf(output, "%-30s: %.6f sec (%.2fx)\n", label, time_sec, speedup);
    write_result(output, sweep, rows, inner, cols, &configs[config_idx], iterations, time_sec, baseline_sec);
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
    tk_matmul_cfg configs[CONFIG_CAPACITY];
    size_t config_count = 0;

    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IJK, 0, 0, 1, 1));
    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_PTHREAD, TK_LOOP_IJK, 0, 0, thread_count, 1));
    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IJK, 1, 0, 1, block_size));
    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_PTHREAD, TK_LOOP_IJK, 1, 0, thread_count, block_size));

    if (tk_sse_available()) {
      add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IKJ, 0, 1, 1, 1));
      add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IKJ, 1, 1, 1, block_size));
      add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_PTHREAD, TK_LOOP_IKJ, 0, 1, thread_count, 1));
      add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_PTHREAD, TK_LOOP_IKJ, 1, 1, thread_count, block_size));
    }

#if ENABLE_OPENMP
    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_OPENMP, TK_LOOP_IJK, 0, 0, thread_count, 1));
    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_OPENMP, TK_LOOP_IJK, 1, 0, thread_count, block_size));

    if (tk_sse_available()) {
      add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_OPENMP, TK_LOOP_IKJ, 0, 1, thread_count, 1));
      add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_OPENMP, TK_LOOP_IKJ, 1, 1, thread_count, block_size));
    }
#endif

    bench_run_case(output, "matrix_size", matrix_size, matrix_size, matrix_size, iterations, configs, config_count);
  }
}

static void bench_thread_count_sweep(FILE *output) {
  const size_t matrix_size = 512;
  const size_t iterations = 10;
  const size_t thread_counts[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const size_t block_size = 32;

  for (size_t thread_idx = 0; thread_idx < sizeof(thread_counts) / sizeof(thread_counts[0]); ++thread_idx) {
    size_t thread_count = thread_counts[thread_idx];
    tk_matmul_cfg configs[CONFIG_CAPACITY];
    size_t config_count = 0;

    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_PTHREAD, TK_LOOP_IJK, 0, 0, thread_count, 1));
    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_PTHREAD, TK_LOOP_IJK, 1, 0, thread_count, block_size));

    if (tk_sse_available()) {
      add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_PTHREAD, TK_LOOP_IKJ, 0, 1, thread_count, 1));
      add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_PTHREAD, TK_LOOP_IKJ, 1, 1, thread_count, block_size));
    }

#if ENABLE_OPENMP
    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_OPENMP, TK_LOOP_IJK, 0, 0, thread_count, 1));
    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_OPENMP, TK_LOOP_IJK, 1, 0, thread_count, block_size));

    if (tk_sse_available()) {
      add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_OPENMP, TK_LOOP_IKJ, 0, 1, thread_count, 1));
      add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_OPENMP, TK_LOOP_IKJ, 1, 1, thread_count, block_size));
    }
#endif

    bench_run_case(output, "thread_count", matrix_size, matrix_size, matrix_size, iterations, configs, config_count);
  }
}

static void bench_block_size_sweep(FILE *output) {
  const size_t matrix_size = 512;
  const size_t iterations = 10;
  const size_t thread_count = 4;
  const size_t block_sizes[] = {2, 4, 8, 16, 32, 64, 128};

  for (size_t block_idx = 0; block_idx < sizeof(block_sizes) / sizeof(block_sizes[0]); ++block_idx) {
    size_t block_size = block_sizes[block_idx];
    tk_matmul_cfg configs[CONFIG_CAPACITY];
    size_t config_count = 0;

    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IJK, 1, 0, 1, block_size));
    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_PTHREAD, TK_LOOP_IJK, 1, 0, thread_count, block_size));
    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IKJ, 1, 0, 1, block_size));
    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_PTHREAD, TK_LOOP_IKJ, 1, 0, thread_count, block_size));

    if (tk_sse_available()) {
      add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_SINGLE, TK_LOOP_IKJ, 1, 1, 1, block_size));
      add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_PTHREAD, TK_LOOP_IKJ, 1, 1, thread_count, block_size));
    }

#if ENABLE_OPENMP
    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_OPENMP, TK_LOOP_IJK, 1, 0, thread_count, block_size));
    add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_OPENMP, TK_LOOP_IKJ, 1, 0, thread_count, block_size));

    if (tk_sse_available()) { add_config(configs, &config_count, tk_matmul_cfg_new(TK_BACKEND_OPENMP, TK_LOOP_IKJ, 1, 1, thread_count, block_size)); }
#endif

    bench_run_case(output, "block_size", matrix_size, matrix_size, matrix_size, iterations, configs, config_count);
  }
}

void tk_bench_default_suite(const char *output_csv) {
  const char *path = output_csv ? output_csv : "results/data/benchmark_results.csv";
  FILE *output = fopen(path, "w");
  if (!output) {
    TK_LOGE("cannot open benchmark output '%s': %s", path, strerror(errno));
    return;
  }

  fprintf(output, "sweep,rows,inner,cols,backend,loop_order,use_blocking,use_simd,num_threads,block_size,iterations,time_sec,"
                  "speedup_vs_baseline,label\n");

  bench_matrix_size_sweep(output);
  bench_thread_count_sweep(output);
  bench_block_size_sweep(output);

  fclose(output);
  printf("\nWrote benchmark results to %s\n", path);
}
