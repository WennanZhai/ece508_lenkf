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


static void sb_toe_r_init(sb_toe_r *A);
static void sb_toe_r_init_r(sb_toe_r *A, int depth, int *count);


static void sb_toe_r_init(sb_toe_r *A) {
  int count;

  assert(A);

  count = 1;
  sb_toe_r_init_r(A, 0, &count);
}

static void sb_toe_r_init_r(sb_toe_r *A, int depth, int *count) {
  int i;

  assert(A);
  assert(depth >= 0);
  assert(depth < A->dim->rank);
  assert(count);

  if (depth == A->dim->rank - 1) {
    for (i = 0; i < A->dim->n_phy[depth]; i++) {
      A->ptr.v[i] = *count;
      (*count)++;
    }
  }
  else {
    for (i = 0; i < A->dim->n_phy[depth]; i++) {
      sb_toe_r_init_r(A->ptr.A_block[i], depth + 1, count);
    }
  }
}


int main(int argc, char **argv) {
  s_toe *A;
  s_toe_it it_A;
  int i;
  s_toe *B;
  s_toe *empty;
  s_toe_it it_empty;
  FILE *fid;
  int sizeof_elem;
  int n_block, k, k_nnz;
  sb_toe *C;
  sb_toe_it it_C;
  sb_toe *D;
  sb_toe_it it_D;
  int r;

  sb_toe_r *X;
  sb_toe_r_it *X_it;
  sb_toe_r *Y;
  sparse_coo *X_coo;
  int rank;
  int *n_phy;
  int *n;
  int j_prev;
  elem v;


  A = s_toe_create(5, 3);

  A->v_row0[0] = 5;
  A->v_row0[1] = 4;
  A->v_row0[2] = 3;

  s_toe_export("/tmp/A", A);

  printf("A:\n");
  s_toe_printf(A);
  printf("\n");

  /* Answer:

     A:
     +5.000000 +4.000000 +3.000000 +0.000000 +0.000000
     +4.000000 +5.000000 +4.000000 +3.000000 +0.000000
     +3.000000 +4.000000 +5.000000 +4.000000 +3.000000
     +0.000000 +3.000000 +4.000000 +5.000000 +4.000000
     +0.000000 +0.000000 +3.000000 +4.000000 +5.000000
   */

  printf("A (non-zero):\n");
  for (i = 0; i < A->n; i++) {
    s_toe_iterator(A, &it_A, i);

    while (s_toe_nz_it_has_next(&it_A)) {
      printf("(");
      printf_elem(s_toe_nz_it_next(&it_A));
      printf(", %d) ", it_A.j);
    }
    printf("\n");
  }

  /* Answer:

     A (non-zero):
     (+5.000000, 0) (+4.000000, 1) (+3.000000, 2)
     (+4.000000, 0) (+5.000000, 1) (+4.000000, 2) (+3.000000, 3)
     (+3.000000, 0) (+4.000000, 1) (+5.000000, 2) (+4.000000, 3) (+3.000000, 4)
     (+3.000000, 1) (+4.000000, 2) (+5.000000, 3) (+4.000000, 4)
     (+3.000000, 2) (+4.000000, 3) (+5.000000, 4)
  */


  B = s_toe_create(5, 2);

  B->v_row0[0] = 2;
  B->v_row0[1] = 1;

  printf("\n");
  printf("B:\n");
  s_toe_printf(B);

  /* Answer:

     B:
     +2.000000 +1.000000 +0.000000 +0.000000 +0.000000
     +1.000000 +2.000000 +1.000000 +0.000000 +0.000000
     +0.000000 +1.000000 +2.000000 +1.000000 +0.000000
     +0.000000 +0.000000 +1.000000 +2.000000 +1.000000
     +0.000000 +0.000000 +0.000000 +1.000000 +2.000000
   */


  empty = s_toe_create(5, 0);

  printf("\n");
  printf("empty:\n");
  s_toe_printf(empty);

  printf("\n");
  printf("empty (non-zero):\n");
  printf("<start>\n");
  for (i = 0; i < empty->n; i++) {
    s_toe_iterator(empty, &it_empty, i);

    while (s_toe_nz_it_has_next(&it_empty)) {
      printf_elem_s(s_toe_nz_it_next(&it_empty));
    }
    printf("\n");
  }
  printf("<stop>\n");


  /* Answer:
     empty (non-zero):
     <start>





     <stop>
   */


  fid = fopen("/tmp/C", "w");
  assert(fid);

  n_block = 5;
  k = 3;
  k_nnz = 2;

  sizeof_elem = sizeof(elem);
  r = fwrite(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&k, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&n_block, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&k_nnz, sizeof(int), 1, fid);
  assert(r == 1);

  s_toe_export_fid(fid, A);
  s_toe_export_fid(fid, B);
  s_toe_export_fid(fid, empty);

  fclose(fid);

  C = sb_toe_import("/tmp/C");

  printf("\n");
  printf("C:\n");
  sb_toe_printf(C);

  /* Answer:

     C:
     +5.000000 +4.000000 +3.000000 +0.000000 +0.000000 +2.000000 +1.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000
     +4.000000 +5.000000 +4.000000 +3.000000 +0.000000 +1.000000 +2.000000 +1.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000
     +3.000000 +4.000000 +5.000000 +4.000000 +3.000000 +0.000000 +1.000000 +2.000000 +1.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000
     +0.000000 +3.000000 +4.000000 +5.000000 +4.000000 +0.000000 +0.000000 +1.000000 +2.000000 +1.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000
     +0.000000 +0.000000 +3.000000 +4.000000 +5.000000 +0.000000 +0.000000 +0.000000 +1.000000 +2.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000
     +2.000000 +1.000000 +0.000000 +0.000000 +0.000000 +5.000000 +4.000000 +3.000000 +0.000000 +0.000000 +2.000000 +1.000000 +0.000000 +0.000000 +0.000000
     +1.000000 +2.000000 +1.000000 +0.000000 +0.000000 +4.000000 +5.000000 +4.000000 +3.000000 +0.000000 +1.000000 +2.000000 +1.000000 +0.000000 +0.000000
     +0.000000 +1.000000 +2.000000 +1.000000 +0.000000 +3.000000 +4.000000 +5.000000 +4.000000 +3.000000 +0.000000 +1.000000 +2.000000 +1.000000 +0.000000
     +0.000000 +0.000000 +1.000000 +2.000000 +1.000000 +0.000000 +3.000000 +4.000000 +5.000000 +4.000000 +0.000000 +0.000000 +1.000000 +2.000000 +1.000000
     +0.000000 +0.000000 +0.000000 +1.000000 +2.000000 +0.000000 +0.000000 +3.000000 +4.000000 +5.000000 +0.000000 +0.000000 +0.000000 +1.000000 +2.000000
     +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +2.000000 +1.000000 +0.000000 +0.000000 +0.000000 +5.000000 +4.000000 +3.000000 +0.000000 +0.000000
     +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +1.000000 +2.000000 +1.000000 +0.000000 +0.000000 +4.000000 +5.000000 +4.000000 +3.000000 +0.000000
     +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +1.000000 +2.000000 +1.000000 +0.000000 +3.000000 +4.000000 +5.000000 +4.000000 +3.000000
     +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +1.000000 +2.000000 +1.000000 +0.000000 +3.000000 +4.000000 +5.000000 +4.000000
     +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000
   */


  printf("\n");
  printf("C (non-zero):\n");
  for (i = 0; i < C->n; i++) {
    sb_toe_iterator(C, &it_C, i);

    while (sb_toe_nz_it_has_next(&it_C)) {
      printf("(");
      printf_elem(sb_toe_nz_it_next(&it_C));
      printf(", %d)", it_C.j);
    }
    printf("\n");
  }


  /* Answer:

     C (non-zero):
     (+5.000000, 0)(+4.000000, 1)(+3.000000, 2)(+2.000000, 5)(+1.000000, 6)
     (+4.000000, 0)(+5.000000, 1)(+4.000000, 2)(+3.000000, 3)(+1.000000, 5)(+2.000000, 6)(+1.000000, 7)
     (+3.000000, 0)(+4.000000, 1)(+5.000000, 2)(+4.000000, 3)(+3.000000, 4)(+1.000000, 6)(+2.000000, 7)(+1.000000, 8)
     (+3.000000, 1)(+4.000000, 2)(+5.000000, 3)(+4.000000, 4)(+1.000000, 7)(+2.000000, 8)(+1.000000, 9)
     (+3.000000, 2)(+4.000000, 3)(+5.000000, 4)(+1.000000, 8)(+2.000000, 9)
     (+2.000000, 0)(+1.000000, 1)(+5.000000, 5)(+4.000000, 6)(+3.000000, 7)(+2.000000, 10)(+1.000000, 11)
     (+1.000000, 0)(+2.000000, 1)(+1.000000, 2)(+4.000000, 5)(+5.000000, 6)(+4.000000, 7)(+3.000000, 8)(+1.000000, 10)(+2.000000, 11)(+1.000000, 12)
     (+1.000000, 1)(+2.000000, 2)(+1.000000, 3)(+3.000000, 5)(+4.000000, 6)(+5.000000, 7)(+4.000000, 8)(+3.000000, 9)(+1.000000, 11)(+2.000000, 12)(+1.000000, 13)
     (+1.000000, 2)(+2.000000, 3)(+1.000000, 4)(+3.000000, 6)(+4.000000, 7)(+5.000000, 8)(+4.000000, 9)(+1.000000, 12)(+2.000000, 13)(+1.000000, 14)
     (+1.000000, 3)(+2.000000, 4)(+3.000000, 7)(+4.000000, 8)(+5.000000, 9)(+1.000000, 13)(+2.000000, 14)
     (+2.000000, 5)(+1.000000, 6)(+5.000000, 10)(+4.000000, 11)(+3.000000, 12)
     (+1.000000, 5)(+2.000000, 6)(+1.000000, 7)(+4.000000, 10)(+5.000000, 11)(+4.000000, 12)(+3.000000, 13)
     (+1.000000, 6)(+2.000000, 7)(+1.000000, 8)(+3.000000, 10)(+4.000000, 11)(+5.000000, 12)(+4.000000, 13)(+3.000000, 14)
     (+1.000000, 7)(+2.000000, 8)(+1.000000, 9)(+3.000000, 11)(+4.000000, 12)(+5.000000, 13)(+4.000000, 14)
     (+1.000000, 8)(+2.000000, 9)(+3.000000, 12)(+4.000000, 13)(+5.000000, 14)
  */

  fid = fopen("/tmp/D", "w");
  assert(fid);

  n_block = 5;
  k = 2;
  k_nnz = 0;

  sizeof_elem = sizeof(elem);
  r = fwrite(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&k, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&n_block, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&k_nnz, sizeof(int), 1, fid);
  assert(r == 1);

  s_toe_export_fid(fid, empty);
  s_toe_export_fid(fid, empty);

  fclose(fid);

  D = sb_toe_import("/tmp/D");

  printf("\n");
  printf("D:\n");
  sb_toe_printf(D);

  /* Answer:

     D:
     +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000
     +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000
     +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000
     +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000
     +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000
     +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000
     +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000
     +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000
     +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000
     +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000000
   */


  printf("\n");
  printf("D (non-zero):\n");
  printf("<start>\n");
  for (i = 0; i < D->n; i++) {
    sb_toe_iterator(D, &it_D, i);

    while (sb_toe_nz_it_has_next(&it_D)) {
      printf("(");
      printf_elem(sb_toe_nz_it_next(&it_D));
      printf(", %d)", it_D.j);
    }
    printf("\n");
  }
  printf("<stop>\n");


  /* Answer:

     D (non-zero):
     <start>










     <stop>
  */


  s_toe_destroy(&A);
  s_toe_destroy(&B);
  s_toe_destroy(&empty);

  sb_toe_destroy(&C);
  sb_toe_destroy(&D);



  /****************************************************************************/
  /* sb_toe_r */

  //rank = 2;
  rank = 3;

  n_phy = malloc(rank * sizeof(int));
  assert(n_phy);

  n = malloc(rank * sizeof(int));
  assert(n);

  n_phy[0] = 2;
  n_phy[1] = 3;
  n_phy[2] = 4;

  n[0] = 4;
  n[1] = 6;
  n[2] = 5;

  X = sb_toe_r_create(rank, n_phy, n);


  sb_toe_r_init(X);

  fid = fopen("/tmp/X1", "w");
  assert(fid);

  printf("\n");
  printf("X:\n");
  sb_toe_r_printf(X);
  sb_toe_r_fprintf(fid, X);
  fclose(fid);


  X_coo = sb_toe_r_convert_coo(X);
  sparse_coo_export("/tmp/X_coo", X_coo);
  sparse_coo_destroy(&X_coo);

  X_it = sb_toe_r_nz_it_create(X);

  fid = fopen("/tmp/X2", "w");
  assert(fid);

  printf("\n");
  for (i = 0; i < X->dim->N[0]; i++) {
    sb_toe_r_nz_it_init(X_it, i);
    j_prev = -1;
    while (sb_toe_r_nz_it_has_next(X_it)) {
      v = sb_toe_r_nz_it_next(X_it);

      for (k = j_prev; k < X_it->j - 1; k++) {
	printf_elem_s(0);
	fprintf_elem_s(fid, 0);
      }

      j_prev = X_it->j;

      printf_elem_s(v);

      /*
      printf("(");
      printf_elem_s(v);
      printf("\b, %d) ", X_it->j);
      */

      fprintf_elem_s(fid, v);
    }

    for (k = X_it->j_last; k < X_it->A->dim->N[0]; k++) {
      printf_elem_s(0);
      fprintf_elem_s(fid, 0);
    }

    printf("\n");
    fprintf(fid, "\n");
  }

  fclose(fid);

  sb_toe_r_nz_it_destroy(&X_it);

  sb_toe_r_export("/tmp/X_toeplitz_new_test", X);

  Y = sb_toe_r_import("/tmp/X_toeplitz_new_test");

  printf("\n");
  printf("Y:\n");
  sb_toe_r_printf(Y);

  sb_toe_r_destroy(&X);
  sb_toe_r_destroy(&Y);


  /*
  X = sb_toe_r_import("/tmp/test");

  fid = fopen("/tmp/test_out", "w");
  assert(fid);

  X_it = sb_toe_r_it_create(X);

  for (i = 0; i < X->dim->N[0]; i++) {
    sb_toe_r_it_init(X_it, i);

    while(sb_toe_r_it_has_next(X_it)) {
      fprintf_elem_s(fid, sb_toe_r_it_next(X_it));
    }
    fprintf(fid, "\n");
  }

  sb_toe_r_it_destroy(&X_it);

  fclose(fid);

  fid = fopen("/tmp/test2_out", "w");

  X_it = sb_toe_r_nz_it_create(X);

  for (i = 0; i < X->dim->N[0]; i++) {
    sb_toe_r_nz_it_init(X_it, i);
    j_prev = -1;
    while (sb_toe_r_nz_it_has_next(X_it)) {
      v = sb_toe_r_nz_it_next(X_it);

      for (k = j_prev; k < X_it->j - 1; k++) {
	fprintf_elem_s(fid, 0);
      }

      j_prev = X_it->j;

      fprintf_elem_s(fid, v);
    }

    for (k = X_it->j_last; k < X_it->A->dim->N[0]; k++) {
      fprintf_elem_s(fid, 0);
    }

    fprintf(fid, "\n");
  }

  sb_toe_r_nz_it_destroy(&X_it);

  fclose(fid);

  sb_toe_r_destroy(&X);
  */


  return 0;
}
