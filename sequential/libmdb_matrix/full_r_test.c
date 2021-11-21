#ifdef FLOAT_ELEM_TEST
#include "mdb_matrix_s.h"
#elif defined DOUBLE_ELEM_TEST
#include "mdb_matrix_d.h"
#else
#error 0
#endif


int main(void) {
  full_r *A, *B;
  int mA, nA, mB, nB;
  int i, j, index;
  
  mA = 3;
  nA = 2;

  mB = 5;
  nB = mB;
  
  A = full_r_create(mA, nA);
  B = full_r_create(mB, nB);
  
  index = 0;
  for (i = 0; i < mA; i++) {
    for (j = 0; j < nA; j++) {
      A->v_vector[index] = index;
      index++;
    }
  }

  index = 0;
  for (i = 0; i < mB; i++) {
    for (j = 0; j < nB; j++) {
      B->v_vector[index] = index;
      index++;
    }
  }
  
  full_r_printf(A);
  /* Answer:

     +0.000000 +1.000000 
     +2.000000 +3.000000 
     +4.000000 +5.000000 
  */

  printf("\n");
  printf_elem_n(full_r_trace(B));
  /* Answer:

     +60.000000
  */
  
  full_r_destroy(&A);
  full_r_destroy(&B);
  
  return 0;
}
