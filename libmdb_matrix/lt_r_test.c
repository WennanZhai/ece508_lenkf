#ifdef FLOAT_ELEM_TEST
#include "mdb_matrix_s.h"
#elif defined DOUBLE_ELEM_TEST
#include "mdb_matrix_d.h"
#else
#error 0
#endif


int main(int argc, char **argv) {
  lt_r *A;
  int m, n;
  int i;

#ifndef VECLIB
  full_r *A2;
#endif

  /* Case m < n */

  m = 4;
  n = 6;

  A = lt_r_create(m, n);

  for (i = 0; i < A->N; i++) {
    A->v[i] = i+1;
  }

  printf("A (m < n):\n");
  lt_r_printf(A);

  lt_r_destroy(&A);


  /* Case m > n */

  m = 6;
  n = 4;

  A = lt_r_create(m, n);

  for (i = 0; i < A->N; i++) {
    A->v[i] = i+1;
  }

  printf("\n");
  printf("A (m > n):\n");
  lt_r_printf(A);

  lt_r_destroy(&A);


  /* Case m == n */

  m = 4;
  n = 4;

  A = lt_r_create(m, n);

  A->v[0] = 103;
  A->v[1] = 86;
  A->v[2] = 94;
  A->v[3] = 74;
  A->v[4] = 72;
  A->v[5] = 60;
  A->v[6] = 80;
  A->v[7] = 72;
  A->v[8] = 68;
  A->v[9] = 100;

  printf("\n");
  printf("A (m == n):\n");
  lt_r_printf(A);

#ifndef VECLIB
  A2 = full_r_create(m, n);
  full_r_set0(A2);

  A2->v[0][0] = 103;
  A2->v[0][1] = 86;
  A2->v[0][2] = 74;
  A2->v[0][3] = 80;
  A2->v[1][1] = 94;
  A2->v[1][2] = 72;
  A2->v[1][3] = 72;
  A2->v[2][2] = 60;
  A2->v[2][3] = 68;
  A2->v[3][3] = 100;

  A2->v[3][0] = -999;

  printf("\n");
  printf("A2:\n");
  full_r_printf(A2);

  epotrf(LAPACK_ROW_MAJOR, 'U', A2->m, A2->v_vector, A2->n);

  printf("\n");
  printf("chol(A2):\n");
  full_r_printf(A2);

  full_r_destroy(&A2);
#endif

  lt_r_destroy(&A);

  return 0;
}
