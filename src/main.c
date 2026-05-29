#include "bench.h"
#include "test.h"

#include <stdio.h>
#include <string.h>

static void print_usage(const char *program) { fprintf(stderr, "usage: %s [test|bench|all]\n", program); }

static int run_tests(void) {
  if (!test_matmul_correctness()) {
    fprintf(stderr, "correctness tests failed\n");
    return 1;
  }

  return 0;
}

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "test";

  if (argc > 2) {
    print_usage(argv[0]);
    return 1;
  }

  if (strcmp(mode, "test") == 0) {
    return run_tests();
  }

  if (strcmp(mode, "bench") == 0) {
    bench_run_default_suite("benchmark_results.csv");
    return 0;
  }

  if (strcmp(mode, "all") == 0) {
    if (run_tests() != 0) {
      return 1;
    }
    bench_run_default_suite("benchmark_results.csv");
    return 0;
  }

  print_usage(argv[0]);
  return 1;
}
