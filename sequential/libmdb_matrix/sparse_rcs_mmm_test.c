#ifdef FLOAT_ELEM_TEST
#include "mdb_matrix_s.h"
#elif defined DOUBLE_ELEM_TEST
#include "mdb_matrix_d.h"
#else
#error 0
#endif


int main(int argc, char **argv) {
  sparse_rcs *A, *B_T;
  full_r *C;

  A = sparse_rcs_import("/tmp/A");
  B_T = sparse_rcs_import("/tmp/B_T");

  C = full_r_create(A->m, B_T->m);

  sparse_rcs_mmm(A, B_T, C);

  full_r_printf(C);

  full_r_export("/tmp/C", C);
  
  sparse_rcs_destroy(&A);
  sparse_rcs_destroy(&B_T);
  full_r_destroy(&C);

  return 0;
}
