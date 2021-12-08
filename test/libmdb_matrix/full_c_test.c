#ifdef FLOAT_ELEM_TEST
#include "mdb_matrix_s.h"
#elif defined DOUBLE_ELEM_TEST
#include "mdb_matrix_d.h"
#else
#error 0
#endif


int main(void) {
  full_c *A, *B;
  int mA, nA, mB, nB;
  int i, j, index;
  
  mA = 3;
  nA = 2;

  mB = 5;
  nB = mB;
  
  A = full_c_create(mA, nA);
  B = full_c_create(mB, nB);
  
  index = 0;
  for (j = 0; j < nA; j++) {
    for (i = 0; i < mA; i++) {
      A->v_vector[index] = index;
      index++;
    }
  }

  index = 0;
  for (j = 0; j < nB; j++) {
    for (i = 0; i < mB; i++) {
      B->v_vector[index] = index;
      index++;
    }
  }

  printf("A:\n");
  full_c_printf(A);
  /* Answer:

     +0.000000 +3.000000 
     +1.000000 +4.000000 
     +2.000000 +5.000000 
  */

  printf("\nB:\n");
  full_c_printf(B);
  /* Answer:

    +0.000000 +5.000000 +10.000000 +15.000000 +20.000000 
    +1.000000 +6.000000 +11.000000 +16.000000 +21.000000 
    +2.000000 +7.000000 +12.000000 +17.000000 +22.000000 
    +3.000000 +8.000000 +13.000000 +18.000000 +23.000000 
    +4.000000 +9.000000 +14.000000 +19.000000 +24.000000 
  */
  
  printf("\ntrace(B):\n");
  printf_elem_n(full_c_trace(B));
  /* Answer:

     +60.000000
  */
  
  full_c_destroy(&A);
  full_c_destroy(&B);
  
  return 0;
}
