#ifdef FLOAT_ELEM_TEST
#include "mdb_matrix_s.h"
#elif defined DOUBLE_ELEM_TEST
#include "mdb_matrix_d.h"
#else
#error 0
#endif


zfull_r *matrix_init(int m, int n) {
  zfull_r *A;
  int i;
  int j;
  
  A = zfull_r_create(m, n);

  for (i = 0; i < A->m; i++) {
    for (j = 0; j < A->n; j++) {
      A->v[i][j][0] = i*n + j;
      A->v[i][j][1] = i*n + j;
    }
  }

  return A;
}

int main(int argc, char **argv) {
  zfull_r *A, *B, *C;

  A = matrix_init(2, 2);
  B = matrix_init(2, 3);
  C = zfull_r_create(2, 3);
  
  printf("A:\n");
  zfull_r_printf(A);
  /* Result:
     
    (+0.000000 +0.000000) (+1.000000 +1.000000) 
    (+2.000000 +2.000000) (+3.000000 +3.000000) 
   */
  
  printf("\nB:\n");
  zfull_r_printf(B);
  /* Result:

     (+0.000000 +0.000000) (+1.000000 +1.000000) (+2.000000 +2.000000) 
     (+3.000000 +3.000000) (+4.000000 +4.000000) (+5.000000 +5.000000) 
  */

  zfull_r_mmm(A, B, C);

  printf("\nC:\n");
  zfull_r_printf(C);
  /* Result:

     (+0.000000 +6.000000) (+0.000000 +8.000000) (+0.000000 +10.000000) 
     (+0.000000 +18.000000) (+0.000000 +28.000000) (+0.000000 +38.000000) 
  */

  zfull_r_export("/tmp/C", C);
  
  zfull_r_destroy(&A);
  zfull_r_destroy(&B);
  zfull_r_destroy(&C);
  
  return 0;
}
