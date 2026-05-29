#include "bench.h"
#include "test.h"

#include <stdio.h>

int main(void) {
  if (!test_matmul_correctness()) {
    fprintf(stderr, "correctness tests failed\n");
    return 1;
  }

  bench_run_default_suite("benchmark_results.csv");
  return 0;
}
