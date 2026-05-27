#ifndef TEST_H
#define TEST_H

#include "matmul.h"
#include "matrix.h"

#include <stddef.h>

#define COLOR_RED "\x1b[31m"
#define COLOR_GREEN "\x1b[32m"
#define COLOR_YELLOW "\x1b[33m"
#define COLOR_BLUE "\x1b[34m"
#define COLOR_MAGENTA "\x1b[35m"
#define COLOR_CYAN "\x1b[36m"
#define COLOR_RESET "\x1b[0m"
#define CHECK_MARK "\u2713"

int matrix_equal(const Matrix *a, const Matrix *b, double eps);
void test_matmul_correctness(void);

#endif // TEST_H
