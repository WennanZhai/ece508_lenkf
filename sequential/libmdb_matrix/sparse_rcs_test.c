#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#ifdef FLOAT_ELEM_TEST
#include "mdb_matrix_s.h"
#elif defined DOUBLE_ELEM_TEST
#include "mdb_matrix_d.h"
#else
#error 0
#endif


static void init_A(sparse_rcs *A);
static void init_x(vector *x);


void init_A(sparse_rcs *A) {
  int i;
  
  assert(A);

  A->N = A->m * A->n;

  A->v = malloc(sizeof(elem) * A->N);
  assert(A->v);

  A->j = malloc(sizeof(int) * A->N);
  assert(A->j);
  
  for (i = 0; i < A->N; i++) {
    A->v[i] = i + 1;
    A->j[i] = i % A->n;
  }

  for (i = 0; i <= A->m; i++) {
    A->r[i] = i * A->n;
  }
}

void init_x(vector *x) {
  int i;
  
  assert(x);

  for (i = 0; i < x->n; i++) {
    x->v[i] = i+1;
  }
}

int main(void) {
  sparse_rcs *A;
  /*sparse_rcs *B;*/

  vector *x, *y;
  const int m = 4;
  const int n = 3;

  sparse_ccs *B;
  sparse_rcs *C;
  
  A = sparse_rcs_create(m, n, 0);
  x = vector_create(n);
  y = vector_create(m);

  init_A(A);
  sparse_rcs_printf(A);

  init_x(x);
  printf("\n");
  vector_printf(x);

  sparse_rcs_mvm(A, x, y);
  printf("\n");
  vector_printf(y);
  /* Answer: 
     +14.000000 +32.000000 +50.000000 +68.000000
  */
  
  sparse_rcs_mvm_blas(A, x, y);
  printf("\n");
  vector_printf(y);

  sparse_rcs_export3("/tmp/v_test", "/tmp/j_test", "/tmp/r_test", A);

  /*
  B = sparse_rcs_import3("/tmp/v_test2", "/tmp/j_test2", "/tmp/r_test2");
  printf("\n");
  sparse_rcs_printf(B);

  sparse_rcs_destroy(&B);
  */

  B = sparse_ccs_create(A->n, A->m);
  B->v = A->v;
  B->i = A->j;
  B->c = A->r;
  B->N = A->N;

  printf("\n");
  sparse_ccs_printf(B);
  free(B);

  C = sparse_rcs_transpose(A);
  printf("\n");
  sparse_rcs_printf(C);
  
  sparse_rcs_destroy(&A);
  sparse_rcs_destroy(&C);
  vector_destroy(&x);
  vector_destroy(&y);

  return 0;
}
