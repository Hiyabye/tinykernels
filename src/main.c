#include "bench.h"

// Entry point of the program, runs benchmarks for matrix multiplication
int main(void) {
  // matrix sizes
  run_benchmark(128, 128, 128, 1, 128, 100);
  run_benchmark(256, 256, 256, 1, 256, 100);
  run_benchmark(512, 512, 512, 1, 512, 100);

  // thread count
  run_benchmark(512, 512, 512, 2, 512, 100);
  run_benchmark(512, 512, 512, 4, 512, 100);
  run_benchmark(512, 512, 512, 8, 512, 100);

  // block size
  run_benchmark(512, 512, 512, 1, 32, 100);
  run_benchmark(512, 512, 512, 1, 64, 100);
  run_benchmark(512, 512, 512, 1, 128, 100);

  return 0;
}
