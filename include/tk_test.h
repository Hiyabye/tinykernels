#ifndef TK_TEST_H
#define TK_TEST_H

/* Runs the full test suite (matmul correctness + tokenizer + gguf/model
 * regressions). Returns nonzero on any failure. */
int tk_test_all(void);

#endif /* TK_TEST_H */
