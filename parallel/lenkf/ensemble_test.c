#include <stdio.h>

#include "ensemble.h"
#include "randn.h"
#include "compute.h"

int main(void) {
  const int n = 4;
  const int l = 3;
  ensemble *e, *e2;
  vector *mean;
  int i, j;
  int count;
  
  randn_init();
  
  e = ensemble_create(n, l);
  e2 = ensemble_create(n, l);
  mean = vector_create(n);

  count = 1;
  for (i = 0; i < n; i++) {
    for (j = 0; j < l; j++) {
      e->X->v[i][j] = count;
      e2->X->v[i][j] = count;
      count++;
    }
  }

  ensemble_mean(e, mean);
  vector_printf(mean);
  /* Answer: +2.000000 +5.000000 +8.000000 +11.000000 */
  
  ensemble_subtract_mean(e,  mean);

  printf("\n");
  ensemble_printf(e);
  /* Answer: */
  /* -1.000000 +0.000000 +1.000000 
     -1.000000 +0.000000 +1.000000 
     -1.000000 +0.000000 +1.000000 
     -1.000000 +0.000000 +1.000000  */


  printf("\n");
  ensemble_printf(e2);
  /* Answer:
     +1.000000 +2.000000 +3.000000 
     +4.000000 +5.000000 +6.000000 
     +7.000000 +8.000000 +9.000000 
     +10.000000 +11.000000 +12.000000  */

  
  printf("\n");
  printf_elem_n(ensemble_trace(e2));
  /* Answer:
     +325.000000 */
  
  ensemble_copy(e2, e);
  printf("\n");
  ensemble_printf(e2);
  /* Answer: */
  /* -1.000000 +0.000000 +1.000000 
     -1.000000 +0.000000 +1.000000 
     -1.000000 +0.000000 +1.000000 
     -1.000000 +0.000000 +1.000000  */

  ensemble_destroy(&e);
  ensemble_destroy(&e2);
  vector_destroy(&mean);
  
  randn_exit();
  
  return 0;
}
