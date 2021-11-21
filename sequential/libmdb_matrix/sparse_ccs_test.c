#ifdef FLOAT_ELEM_TEST
#include "mdb_matrix_s.h"
#elif defined DOUBLE_ELEM_TEST
#include "mdb_matrix_d.h"
#else
#error 0
#endif


int main(int argc, char **argv) {
  vector *x, *y;
  sparse_ccs *A;

  const int M = 4;
  const int N = 5;
  
  x = vector_import("/tmp/x");
  A = sparse_ccs_import3_srt("/tmp/v", "/tmp/i", "/tmp/c", M, N);
  y = vector_create(A->m);

  /*sparse_ccs_printf(A);*/
  
  sparse_ccs_mvm(A, x, y);
  vector_printf(y);
  
  vector_destroy(&x);
  sparse_ccs_destroy(&A);
  vector_destroy(&y);
  
  return 0;
}
