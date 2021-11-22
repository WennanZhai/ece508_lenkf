#ifdef FLOAT_ELEM_TEST
#include "mdb_matrix_s.h"
#elif defined DOUBLE_ELEM_TEST
#include "mdb_matrix_d.h"
#else
#error 0
#endif


int main(int argc, char **argv) {
  lt_c *A;
  full_c *B;

  char *A_filename = "/tmp/A";
  char *B_filename = "/tmp/B";

  A = lt_c_import(A_filename);
  B = full_c_import(B_filename);

  printf("A:\n");
  lt_c_printf(A);

  printf("\n");
  printf("B:\n");
  full_c_printf(B);

  printf("\nA->n: %d   A->N: %d\n", A->n, A->N);


  lt_c_eppsv(A, B);

  /*eppsv(LACPACK_COL_MAJOR, 'L', A->n, B->n, A->v, B->v_vector, B->m);*/

  /*
  epptrf(LAPACK_COL_MAJOR, 'L', A->n, A->v);

  printf("chol(A):\n");
  lt_c_printf(A);

  printf("\nB->m=%d   B->n=%d\n", B->m, B->n);
  fflush(stdout);

  epptrs(LAPACK_COL_MAJOR, 'L', A->n, B->n, A->v, B->v_vector, B->m);
  */

  printf("A\\B\n");
  full_c_printf(B);

  lt_c_destroy(&A);
  full_c_destroy(&B);

  return 0;
}
