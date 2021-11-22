#ifdef FLOAT_ELEM_TEST
#include "mdb_matrix_s.h"
#elif defined DOUBLE_ELEM_TEST
#include "mdb_matrix_d.h"
#else
#error 0
#endif


int main(int argc, char **argv) {
  sparse_lil *A;
  sparse_lil_row_it A_it;
  int i;
  sparse_rcs *A_rcs;
  
  sparse_rcs *H;
  full_c *B;
  
  A = sparse_lil_create_block_size(4, 6, 3);

  sparse_lil_append(A, 0, 0, 1);
  sparse_lil_append(A, 0, 4, 5);
  sparse_lil_append(A, 0, 2, 3);
  sparse_lil_append(A, 0, 1, 2);
  sparse_lil_append(A, 0, 3, 4);
  sparse_lil_append(A, 0, 5, 6);

  sparse_lil_append(A, 1, 3, 8);
  sparse_lil_append(A, 1, 1, 7);
  
  sparse_lil_append(A, 3, 2, 9);
  
  printf("A:\n");
  sparse_lil_printf(A);

  printf("\n");
  printf("A->N = %d\n", A->N);
	
  printf("\n");
  for (i = 0; i < A->m; i++) {
    sparse_lil_row_it_init(A, i, &A_it);

    while(sparse_lil_row_it_has_next(&A_it)) {
      printf("(");
      printf_elem(sparse_lil_row_it_next(&A_it));
      printf(", %d) ", A_it.j);
    }

    printf("\n");
  }


  A_rcs = sparse_lil_2_rcs(A);

  printf("\n");
  printf("A_rcs:\n");
  sparse_rcs_printf(A_rcs);
  
  sparse_rcs_destroy(&A_rcs);
  
  H = sparse_rcs_create(3, 4, 4);
  B = full_c_create(3, 6);

  H->v[0] = 1;
  H->v[1] = 2;
  H->v[2] = 3;
  H->v[3] = 4;

  H->j[0] = 0;
  H->j[1] = 2;
  H->j[2] = 1;
  H->j[3] = 3;

  H->r[0] = 0;
  H->r[1] = 2;
  H->r[2] = 2;
  H->r[3] = 4;

  printf("\n");
  printf("H:\n");
  sparse_rcs_printf(H);

  full_c_set0(B);
  sparse_lil_mmm(H, A, B);
  
  printf("\n");
  printf("B:\n");
  full_c_printf(B);
  

  sparse_lil_scal(A, 0.5);

  printf("\n");
  printf("A/2:\n");
  sparse_lil_printf(A);
  
  sparse_lil_destroy(&A);
  sparse_rcs_destroy(&H);
  full_c_destroy(&B);
  
  return 0;
}
