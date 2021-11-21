#include <stdio.h>

#include "mdb_matrix_d.h"


int main(int argc, char **argv) {
  int n, k;
  
  printf("sizeof(int) \t\t= %u (bytes)\n",
	 (unsigned int) sizeof(int));

  printf("sizeof(long int) \t= %u (bytes)\n",
	 (unsigned int) sizeof(long int));

  printf("sizeof(float) \t\t= %u (bytes)\n",
	 (unsigned int) sizeof(float));

  printf("sizeof(double) \t\t= %u (bytes)\n",
	 (unsigned int) sizeof(double));

  printf("sizeof(void *) \t\t= %u (bytes)\n",
	 (unsigned int) sizeof(void *));

  n = 4;
  printf("\nfftfreq(n=%d)\n", n);
  for (k = 0; k < n; k++) {
    printf_elem_s(fftfreq(n, k));
  }
  printf("\n");
     
  /* Results:

    fftfreq(n=4)
    +0.000000 +0.250000 -0.500000 -0.250000 

    Matches numpy:

    In [41]: fftfreq(4)
    Out[41]: array([ 0.  ,  0.25, -0.5 , -0.25])

  */
  
  n = 5;
  printf("\nfftfreq(n=%d)\n", n);
  for (k = 0; k < n; k++) {
    printf_elem_s(fftfreq(n, k));
  }
  printf("\n");

  /* Result:

     fftfreq(n=5)
     +0.000000 +0.200000 +0.400000 -0.400000 -0.200000 

     Matches numpy:

     In [42]: fftfreq(5)
     Out[42]: array([ 0. ,  0.2,  0.4, -0.4, -0.2])

  */

  n = 4;
  printf("\nfftshift(fftfreq(n=%d))\n", n);
  for (k = 0; k < n; k++) {
    printf_elem_s(fftfreq(n, fftshift(n, k)));
  }
  printf("\n");

  /* Result:

     fftshift(fftfreq(n=4))
     -0.500000 -0.250000 +0.000000 +0.250000

     Matches numpy:

     In [43]: fftshift(fftfreq(4))
     Out[43]: array([-0.5 , -0.25,  0.  ,  0.25])

  */

  n = 5;
  printf("\nfftshift(fftfreq(n=%d))\n", n);
  for (k = 0; k < n; k++) {
    printf_elem_s(fftfreq(n, fftshift(n, k)));
  }
  printf("\n");

  /* Result:

     fftshift(fftfreq(n=5))
     -0.400000 -0.200000 +0.000000 +0.200000 +0.400000 

     Matches numpy:

    In [44]: fftshift(fftfreq(5))
    Out[44]: array([-0.4, -0.2,  0. ,  0.2,  0.4])

  */
      
  return 0;
}
