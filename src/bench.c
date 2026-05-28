#include "bench.h"
#include "matmul.h"
#include "matrix.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void bench_matrix_fill(Matrix *m) {
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
  } else {
    return (values[n / 2 - 1] + values[n / 2]) / 2.0;
  }
}

static double bench_kernel(size_t rows, size_t inner, size_t cols, MatmulConfig cfg, size_t iterations) {
  Matrix a = matrix_new(rows, inner);
  Matrix b = matrix_new(inner, cols);

  if (!a.data || !b.data) {
    matrix_free(&a);
    matrix_free(&b);
    return -1.0;
  }

  bench_matrix_fill(&a);
  bench_matrix_fill(&b);

  double *times = malloc(sizeof(double) * iterations);
  if (!times) {
    matrix_free(&a);
    matrix_free(&b);
    return -1.0;
  }

  double result = -1.0;
  Matrix c = matrix_new(rows, cols);
  if (!c.data) {
    goto cleanup;
  }

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
  matrix_free(&c);
  free(times);
  matrix_free(&a);
  matrix_free(&b);

  return result;
}

static void bench_run_case(size_t rows, size_t inner, size_t cols, size_t threads, size_t block_size, size_t iterations,
                           FILE *result_file, const char *sweep_name) {
  printf("\n[benchmark]\n");
  printf("A: %zux%zu, B: %zux%zu, threads: %zu, block size: %zu, iterations: "
         "%zu\n",
         rows, inner, inner, cols, threads, block_size, iterations);

  if (iterations == 0) {
    fprintf(stderr, "iterations must be greater than zero\n");
    return;
  }

  MatmulConfig ref_ijk_cfg = {
      .kernel = MATMUL_REF_IJK,
      .num_threads = 1,
      .block_size = 1,
  };
  double ref_ijk_median = bench_kernel(rows, inner, cols, ref_ijk_cfg, iterations);

  MatmulConfig seq_ikj_cfg = {
      .kernel = MATMUL_SEQ_IKJ,
      .num_threads = 1,
      .block_size = 1,
  };
  double seq_ikj_median = bench_kernel(rows, inner, cols, seq_ikj_cfg, iterations);

  MatmulConfig seq_blocked_ikj_cfg = {
      .kernel = MATMUL_SEQ_BLOCKED_IKJ,
      .num_threads = 1,
      .block_size = block_size,
  };
  double seq_blocked_ikj_median = bench_kernel(rows, inner, cols, seq_blocked_ikj_cfg, iterations);

  MatmulConfig par_rows_ijk_cfg = {
      .kernel = MATMUL_PAR_ROWS_IJK,
      .num_threads = threads,
      .block_size = 1,
  };
  double par_rows_ijk_median = bench_kernel(rows, inner, cols, par_rows_ijk_cfg, iterations);

  MatmulConfig par_rows_blocked_ikj_cfg = {
      .kernel = MATMUL_PAR_ROWS_BLOCKED_IKJ,
      .num_threads = threads,
      .block_size = block_size,
  };
  double par_rows_blocked_ikj_median = bench_kernel(rows, inner, cols, par_rows_blocked_ikj_cfg, iterations);

  MatmulConfig openmp_ikj_cfg = {
      .kernel = MATMUL_OPENMP_IKJ,
      .num_threads = threads,
      .block_size = 1,
  };
  double openmp_ikj_median = bench_kernel(rows, inner, cols, openmp_ikj_cfg, iterations);

  if (ref_ijk_median < 0.0 || seq_ikj_median < 0.0 || seq_blocked_ikj_median < 0.0 || par_rows_ijk_median < 0.0 ||
      par_rows_blocked_ikj_median < 0.0 || openmp_ikj_median < 0.0) {
    fprintf(stderr, "benchmark failed\n");
    return;
  }

  // sweep,n,threads,block_size,iterations,kernel,time_sec,speedup_vs_ref
  if (result_file) {
    fprintf(result_file, "%s,%zu,%zu,%zu,%zu,%s,%f,%f\n", sweep_name, rows, threads, block_size, iterations,
            matmul_kernel_name(ref_ijk_cfg.kernel), ref_ijk_median, 1.0);
    fprintf(result_file, "%s,%zu,%zu,%zu,%zu,%s,%f,%f\n", sweep_name, rows, threads, block_size, iterations,
            matmul_kernel_name(seq_ikj_cfg.kernel), seq_ikj_median,
            seq_ikj_median > 0.0 ? ref_ijk_median / seq_ikj_median : 0.0);
    fprintf(result_file, "%s,%zu,%zu,%zu,%zu,%s,%f,%f\n", sweep_name, rows, threads, block_size, iterations,
            matmul_kernel_name(par_rows_ijk_cfg.kernel), par_rows_ijk_median,
            par_rows_ijk_median > 0.0 ? ref_ijk_median / par_rows_ijk_median : 0.0);
    fprintf(result_file, "%s,%zu,%zu,%zu,%zu,%s,%f,%f\n", sweep_name, rows, threads, block_size, iterations,
            matmul_kernel_name(seq_blocked_ikj_cfg.kernel), seq_blocked_ikj_median,
            seq_blocked_ikj_median > 0.0 ? ref_ijk_median / seq_blocked_ikj_median : 0.0);
    fprintf(result_file, "%s,%zu,%zu,%zu,%zu,%s,%f,%f\n", sweep_name, rows, threads, block_size, iterations,
            matmul_kernel_name(par_rows_blocked_ikj_cfg.kernel), par_rows_blocked_ikj_median,
            par_rows_blocked_ikj_median > 0.0 ? ref_ijk_median / par_rows_blocked_ikj_median : 0.0);
    fprintf(result_file, "%s,%zu,%zu,%zu,%zu,%s,%f,%f\n", sweep_name, rows, threads, block_size, iterations,
            matmul_kernel_name(openmp_ikj_cfg.kernel), openmp_ikj_median,
            openmp_ikj_median > 0.0 ? ref_ijk_median / openmp_ikj_median : 0.0);
  } else {
    printf("-----------------------------------------------------\n");
    printf("%-30s: %.6f sec\n", matmul_kernel_name(ref_ijk_cfg.kernel), ref_ijk_median);
    printf("%-30s: %.6f sec (%.2fx)\n", matmul_kernel_name(seq_ikj_cfg.kernel), seq_ikj_median,
           seq_ikj_median > 0.0 ? ref_ijk_median / seq_ikj_median : 0.0);
    printf("%-30s: %.6f sec (%.2fx)\n", matmul_kernel_name(par_rows_ijk_cfg.kernel), par_rows_ijk_median,
           par_rows_ijk_median > 0.0 ? ref_ijk_median / par_rows_ijk_median : 0.0);
    printf("%-30s: %.6f sec (%.2fx)\n", matmul_kernel_name(seq_blocked_ikj_cfg.kernel), seq_blocked_ikj_median,
           seq_blocked_ikj_median > 0.0 ? ref_ijk_median / seq_blocked_ikj_median : 0.0);
    printf("%-30s: %.6f sec (%.2fx)\n", matmul_kernel_name(par_rows_blocked_ikj_cfg.kernel),
           par_rows_blocked_ikj_median,
           par_rows_blocked_ikj_median > 0.0 ? ref_ijk_median / par_rows_blocked_ikj_median : 0.0);
    printf("%-30s: %.6f sec (%.2fx)\n", matmul_kernel_name(openmp_ikj_cfg.kernel), openmp_ikj_median,
           openmp_ikj_median > 0.0 ? ref_ijk_median / openmp_ikj_median : 0.0);
    printf("-----------------------------------------------------\n");
  }
}

void bench_run_default_suite(const char *output_csv) {
  FILE *result_file = fopen(output_csv, "w");
  if (!result_file) {
    fprintf(stderr, "failed to open benchmark_results.csv for writing\n");
    return;
  }
  fprintf(result_file, "sweep,n,threads,block_size,iterations,kernel,time_sec,speedup_vs_ref\n");

  // matrix size sweep
  bench_run_case(128, 128, 128, 1, 128, 100, result_file, "matrix_size");
  bench_run_case(256, 256, 256, 1, 256, 100, result_file, "matrix_size");
  bench_run_case(512, 512, 512, 1, 512, 100, result_file, "matrix_size");

  // thread count sweep
  bench_run_case(512, 512, 512, 2, 512, 100, result_file, "thread_count");
  bench_run_case(512, 512, 512, 4, 512, 100, result_file, "thread_count");
  bench_run_case(512, 512, 512, 8, 512, 100, result_file, "thread_count");

  // block size sweep
  bench_run_case(512, 512, 512, 1, 32, 100, result_file, "block_size");
  bench_run_case(512, 512, 512, 1, 64, 100, result_file, "block_size");
  bench_run_case(512, 512, 512, 1, 128, 100, result_file, "block_size");

  fclose(result_file);
}
