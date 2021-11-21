#include <stdio.h>
#include <assert.h>

#ifdef FLOAT_ELEM_TEST
#include "mdb_matrix_s.h"
#elif defined DOUBLE_ELEM_TEST
#include "mdb_matrix_d.h"
#else
#error 0
#endif


int main(void) {
  c_filter *h_c;
  c_filter *x_rs1, *x_rs2;
  rs1_filter *h_rs1;
  rs2_filter *h_rs2;
  r_filter *h_r;
  z_elem H_k;
  elem *e_index;
  int N = 4, i, k;


  h_c = c_filter_create(N);
  h_rs1 = rs1_filter_create(N);
  h_rs2 = rs2_filter_create(N);
  x_rs1 = c_filter_create(h_rs1->n_log);
  x_rs2 = c_filter_create(h_rs2->n_log);
  h_r = r_filter_create(N);

  for (i = 0; i < N; i++) {
#ifdef VECLIB
      h_c->h[i][0] = N - i;
#elif defined OPENBLAS
      h_c->h[i] = N - i;
#else
#error 0
#endif
    h_rs1->h[i] = N - i;
    h_rs2->h[i] = N - i;
  }
  c_filter_set_imag0(h_c);

  for (i = 0; i < x_rs1->n; i++) {
#ifdef VECLIB
      x_rs1->h[i][0] = i;
#elif defined OPENBLAS
      x_rs1->h[i] = i;
#else
#error 0
#endif
  }
  c_filter_set_imag0(x_rs1);

  for (i = 0; i < x_rs2->n; i++) {
#ifdef VECLIB
      x_rs2->h[i][0] = i;
#elif defined OPENBLAS
      x_rs2->h[i] = i;
#else
#error 0
#endif
  }
  c_filter_set_imag0(x_rs2);


  e_index = &(h_r->h[0]);
  for (i = 0; i < N; i++) {
    *e_index = N - i;
    e_index++;
  }

  puts("Test complex impulse reseponse:");
  puts("-------------------------------");
  c_filter_printf(h_c);
  printf("\n");

  c_filter_dft(h_c);
  printf("\n");
  c_filter_printf(h_c);
  printf("\n");

  c_filter_idft(h_c);
  printf("\n");
  c_filter_printf(h_c);
  printf("\n");

  /* Result:

     (+4.000000 +0.000000) (+3.000000 +0.000000) (+2.000000 +0.000000) (+1.000000 +0.000000)

     (+10.000000 +0.000000) (+2.000000 -2.000000) (+2.000000 +0.000000) (+2.000000 +2.000000)

     (+4.000000 +0.000000) (+3.000000 +0.000000) (+2.000000 +0.000000) (+1.000000 +0.000000)


     Matches Matlab:

     >> fft([4 3 2 1])

        ans =  10.0000    2.0000 - 2.0000i   2.0000    2.0000 + 2.0000i
  */


  puts("\n");
  puts("Test real, Type-I symmetric impulse response:");
  puts("---------------------------------------------");
  rs1_filter_printf(h_rs1);
  printf("\n");

  rs1_filter_dft(h_rs1);
  printf("\n");
  rs1_filter_printf(h_rs1);
  printf("\n");

  printf("\n");
  for (k = 0; k < h_rs1->n_log; k++) {
    printf_elem_s(rs1_filter_dft_get(h_rs1, k));
  }
  printf("\n");

  rs1_filter_idft(h_rs1);
  printf("\n");
  rs1_filter_printf(h_rs1);
  printf("\n");

  /* Result:

     +4.000000 +3.000000 +2.000000 +1.000000

     +15.000000 +4.000000 +0.000000 +1.000000

     +15.000000 +4.000000 +0.000000 +1.000000 +0.000000 +4.000000

     +4.000000 +3.000000 +2.000000 +1.000000


     Matches Matlab:

     >> fft([4 3 2 1 2 3])

     ans =  15     4     0     1     0     4

  */


  puts("\n");
  puts("Test real, Type-II symmetric impulse response:");
  puts("----------------------------------------------");
  rs2_filter_printf(h_rs2);
  printf("\n");

  rs2_filter_dft(h_rs2);
  printf("\n");
  rs2_filter_printf(h_rs2);
  printf("\n");

  printf("\n");
  for (k = 0; k < h_rs2->n_log; k++) {
    rs2_filter_dft_get(h_rs2, k, &H_k);
    printf_z_elem_s((const z_elem *) &H_k);
  }
  printf("\n");

  rs2_filter_idft(h_rs2);
  printf("\n");
  rs2_filter_printf(h_rs2);
  printf("\n");

  /* Result:

     +4.000000 +3.000000 +2.000000 +1.000000

     +20.000000 +6.308644 +0.000000 +0.448342

     (+20.000000 +0.000000 ) (+5.828427 +2.414214 ) (+0.000000 +0.000000 ) (+0.171573 +0.414214 ) (+0.000000 +0.000000 ) (+0.171573 -0.414214 ) (+0.000000 +0.000000 ) (+5.828427 -2.414214 )


     Matches Matlab:

     >> fft([4 3 2 1 1 2 3 4])

     ans =

     Columns 1 through 5

     20.0000   5.8284 + 2.4142i   0    0.1716 + 0.4142i   0

     Columns 6 through 8

     0.1716 - 0.4142i    0   5.8284 - 2.4142i

   */


  puts("\n");
  puts("Test filtering with a real, Type-I symmetric impulse response (complex signal):");
  puts("-------------------------------------------------------------------------------");

  c_filter_printf(x_rs1);
  printf("\n");

  c_filter_dft(x_rs1);
  printf("\n");
  c_filter_printf(x_rs1);
  printf("\n");

  if (h_rs1->d == S) {
    rs1_filter_dft(h_rs1);
  }
  printf("\n");
  rs1_filter_printf(h_rs1);
  printf("\n");

  /*rs1_filter_execute_noblas(h_rs1, x);*/
  rs1_filter_execute(h_rs1, x_rs1);
  printf("\n");
  c_filter_printf(x_rs1);
  printf("\n");

  c_filter_idft(x_rs1);
  printf("\n");
  c_filter_printf(x_rs1);
  printf("\n");

  /* Result:

     (+33.000000 +0.000000) (+30.000000 +0.000000) (+33.000000 +0.000000) (+42.000000 +0.000000) (+45.000000 +0.000000) (+42.000000 +0.000000)


     Matches Matlab:

     >> ifft(fft(0:5).*fft([4 3 2 1 2 3]))

     ans =  33    30    33    42    45    42
  */


  puts("\n");
  puts("Test filtering with a real, Type-II symmetric impulse response:");
  puts("---------------------------------------------------------------");

  c_filter_printf(x_rs2);
  printf("\n");

  c_filter_dft(x_rs2);
  printf("\n");
  c_filter_printf(x_rs2);
  printf("\n");

  if (h_rs2->d == S) {
    rs2_filter_dft(h_rs2);
  }

  printf("\n");
  for (k = 0; k < h_rs2->n_log; k++) {
    rs2_filter_dft_get(h_rs2, k, &H_k);
    printf_z_elem_s((const z_elem *) &H_k);
  }
  printf("\n");

  rs2_filter_execute_c(h_rs2, x_rs2);
  printf("\n");
  c_filter_printf(x_rs2);
  printf("\n");

  c_filter_idft(x_rs2);
  printf("\n");
  c_filter_printf(x_rs2);
  printf("\n");

  /* Result:

     (+58.000000 +0.000000) (+54.000000 +0.000000) (+58.000000 -0.000000) (+70.000000 +0.000000) (+82.000000 +0.000000) (+86.000000 +0.000000) (+82.000000 +0.000000) (+70.000000 +0.000000)

     Matches Matlab:

     >> ifft(fft(0:7).*fft([4 3 2 1 1 2 3 4]))

     ans =  58    54    58    70    82    86    82    70
  */


  puts("\n");
  puts("Test real impulse response:");
  puts("---------------------------");

  r_filter_printf(h_r);
  printf("\n");

  r_filter_dft(h_r);

  printf("\n");
  r_filter_printf(h_r);
  printf("\n");

  printf("\n");
  for (k = 0; k < h_r->n_log; k++) {
    r_filter_dft_get(h_r, k, &H_k);
    printf_z_elem_s((const z_elem *) &H_k);
  }
  printf("\n");

  r_filter_idft(h_r);

  printf("\n");
  r_filter_printf(h_r);
  printf("\n");

  /* Result:

     +4.000000 +3.000000 +2.000000 +1.000000

     (+10.000000 +0.000000) (+2.000000 -2.000000) (+2.000000 +0.000000)

     (+10.000000 +0.000000) (+2.000000 -2.000000) (+2.000000 +0.000000) (+2.000000 +2.000000)

     Matches Matlab:

     >> fft([4:-1:1])

     ans =   10   (2 - 2i)   2   (2 + 2i)

   */


  puts("\n");
  puts("Test filtering with a real, Type-II symmetric impulse response (real signal):");
  puts("-----------------------------------------------------------------------------");

  rs2_filter_destroy(&h_rs2);
  r_filter_destroy(&h_r);

  h_r = r_filter_create(8);
  h_rs2 = rs2_filter_create(4);

  e_index = &(h_r->h[0]);
  for (i = 0; i < h_r->n_log; i++) {
    *e_index = i+1;
    e_index++;
  }

  for (i = 0; i < h_rs2->n_phy; i++) {
    h_rs2->h[i] = i+1;
  }

  r_filter_dft(h_r);
  rs2_filter_dft(h_rs2);

  printf("h_r->n_phy=%d\th_r->n_log=%d\n", h_r->n_phy, h_r->n_log);

  printf("\n");
  r_filter_printf(h_r);
  printf("\n");

  printf("\n");
  rs2_filter_printf(h_rs2);
  printf("\n");

  printf("\n");
  for (k = 0; k < h_rs2->n_log; k++) {
    rs2_filter_dft_get(h_rs2, k, &H_k);
    printf_z_elem_s((const z_elem *) &H_k);
  }
  printf("\n");

  rs2_filter_execute_r(h_rs2, h_r);

  printf("\n");
  r_filter_printf(h_r);
  printf("\n");

  r_filter_idft(h_r);

  printf("\n");
  r_filter_printf(h_r);
  printf("\n");

  /* Result:

     +102.000000 +106.000000 +102.000000 +90.000000 +78.000000 +74.000000 +78.000000 +90.000000

     Matches Matlab:

     >> ifft(fft(1:8).*fft([1 2 3 4 4 3 2 1]))

     ans =

     102   106   102    90    78    74    78    90

   */


  c_filter_destroy(&h_c);
  rs1_filter_destroy(&h_rs1);
  rs2_filter_destroy(&h_rs2);
  c_filter_destroy(&x_rs1);
  c_filter_destroy(&x_rs2);
  r_filter_destroy(&h_r);

  fftwe_cleanup();

  return 0;
}
