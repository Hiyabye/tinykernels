#ifndef TEST_H
#define TEST_H

#include "matmul.h"
#include "matrix.h"

#include <stddef.h>

int matrix_equal(const Matrix *a, const Matrix *b, double eps);
void test_matmul_correctness(void);

#endif // TEST_H
