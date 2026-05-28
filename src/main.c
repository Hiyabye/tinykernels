#include "bench.h"
#include "test.h"

int main(void) {
  test_matmul_correctness();
  bench_run_default_suite();

  return 0;
}
