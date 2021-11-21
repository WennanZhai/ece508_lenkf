#include <stdio.h>

#ifdef FLOAT_ELEM_TEST
#include "mdb_matrix_s.h"
#elif defined DOUBLE_ELEM_TEST
#include "mdb_matrix_d.h"
#else
#error 0
#endif


int main(void) {
  s_block_toeplitz *A;
  char A_filename[] = "/tmp/A";
  char A_export_filename[] = "/tmp/A_export";
  sparse_rcs *A_rcs;
  
  int row;

  int i, j;
  
  A = s_block_toeplitz_import(A_filename);

  printf("\n\n");
  s_block_toeplitz_printf_blocks(A);
  printf("\n");

  for (row = 0; row < A->n; row++) {
    printf("A(%d, :)\n", row);
    s_block_toeplitz_printf_row(A, row);
    printf("\n\n");
  }
  
  s_block_toeplitz_export(A_export_filename, A);


  for (i = 0; i < A->n; i++) {
    for (j = 0; j < A->n; j++) {
      printf_elem_s(s_block_toeplitz_get(A, i, j));
    }
    printf("\n");
  }

  A_rcs = s_block_toeplitz_convert(A);
  
  s_block_toeplitz_destroy(&A);

  sparse_rcs_printf(A_rcs);

  sparse_rcs_export("/tmp/A_rcs", A_rcs);
  
  sparse_rcs_destroy(&A_rcs);
  
  return 0;
}
