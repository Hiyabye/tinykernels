#include "test.h"

// Entry point of the program, runs benchmarks for matrix multiplication
int main(void) {
  run_benchmark(200, 200, 200, 10, 100);
  run_benchmark(1000, 1000, 1000, 10, 10);

  return 0;
}
