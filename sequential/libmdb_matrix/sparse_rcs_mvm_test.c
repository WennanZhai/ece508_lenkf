#ifdef FLOAT_ELEM_TEST
#include "mdb_matrix_s.h"
#elif defined DOUBLE_ELEM_TEST
#include "mdb_matrix_d.h"
#else
#error 0
#endif


static void init_x(vector *x);


void init_x(vector *x) {
  int i;

  for (i = 0; i < x->n; i++) {
    x->v[i] = i + 1;
  }
}

int main(void) {
  sparse_rcs *A;
  vector *x, *y, *y_blas;
  int i;
  const int N = 2;

  A = sparse_rcs_import("A");

  x = vector_create(A->n);
  y = vector_create(A->m);
  y_blas = vector_create(A->m);

  init_x(x);

  for (i = 0; i < N; i++) {
    sparse_rcs_mvm(A, x, y);
  }

  for (i = 0; i < N; i++) {
    sparse_rcs_mvm_blas(A, x, y_blas);
  }

  sparse_rcs_destroy(&A);
  vector_destroy(&x);
  vector_destroy(&y);
  vector_destroy(&y_blas);

  return 0;
}
