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
  c_psf_2d *p_c;
  r_psf_2d *p_r;
  r_psf_2d *p_r2;
  rs22_psf_2d *p_rs22;

  int nx = 4;
  int ny = 3;

  fftwe_complex *c_ptr;

  int i, j;


  p_c = c_psf_2d_create(nx, ny);

  c_psf_2d_set0(p_c);

  c_ptr = p_c->h;
  for (i = 0; i < ny; i++) {
    for (j = 0; j < nx; j++) {
#ifdef VECLIB
        p_c->h[i + j*ny][0] = (i+1)*(j + i*nx + 1);
#elif defined OPENBLAS
        p_c->h[i + j*ny] = (i+1)*(j + i*nx + 1);
#else
#error 0
#endif
    }
  }
  c_psf_2d_set_imag0(p_c);

  p_r = r_psf_2d_create(nx, ny);
  r_psf_2d_set0(p_r);

  p_r2 = r_psf_2d_create(nx, ny);
  r_psf_2d_set0(p_r2);

  for (i = 0; i < p_r->ny_phy; i++) {
    for (j = 0; j < p_r->nx_phy; j++) {
      p_r->h[i + j*p_r->ny_phy]  = (i+1)*(j + i*nx + 1);
      p_r2->h[i + j*p_r->ny_phy] = (i+1)*(j + i*nx + 1);
    }
  }

  p_rs22 = rs22_psf_2d_create(nx, ny);

  for (i = 0; i < p_rs22->ny_phy; i++) {
    for (j = 0; j < p_rs22->nx_phy; j++) {
      p_rs22->h[i + j*p_rs22->ny_phy] = (i+j) % p_rs22->nx_phy + 1;
    }
  }


  puts("Test complex psf_2d:");
  puts("--------------------");

  c_psf_2d_printf(p_c);

  c_psf_2d_dft(p_c);

  printf("\n");
  c_psf_2d_printf(p_c);


  /* Result:

     (+188.000000 +0.000000) (-12.000000 +12.000000) (-12.000000 +0.000000) (-12.000000 -12.000000)
     (-79.000000 +64.085876) (+1.267949 -4.732051) (+3.000000 -1.732051) (+4.732051 +1.267949)
     (-79.000000 -64.085876) (+4.732051 -1.267949) (+3.000000 +1.732051) (+1.267949 +4.732051)


     Matches Matlab:

     >> fft2([1:4;(5:8)*2;(9:12)*3])

     ans =

     1.0e+02 *

      1.8800            -0.1200 + 0.1200i  -0.1200            -0.1200 - 0.1200i
     -0.7900 + 0.6409i   0.0127 - 0.0473i   0.0300 - 0.0173i   0.0473 + 0.0127i
     -0.7900 - 0.6409i   0.0473 - 0.0127i   0.0300 + 0.0173i   0.0127 + 0.0473i

  */

  c_psf_2d_idft(p_c);

  printf("\n");
  c_psf_2d_printf(p_c);


  puts("");
  puts("Test real psf_2d:");
  puts("-----------------");

  r_psf_2d_printf(p_r);

  r_psf_2d_dft(p_r);

  printf("\n");
  r_psf_2d_printf(p_r);

  printf("\n");
  r_psf_2d_printf_dft(p_r);

  r_psf_2d_idft(p_r);

  printf("\n");
  r_psf_2d_printf(p_r);

  printf("\n");
  r_psf_2d_printf(p_r2);

  r_psf_2d_dft(p_r);
  r_psf_2d_dft(p_r2);
  r_psf_2d_execute_r(p_r, p_r2);

  r_psf_2d_idft(p_r2);
  printf("\n");
  r_psf_2d_printf(p_r2);


  /* Result:

     +3314.000000 +3340.000000 +3314.000000 +3236.000000
     +4242.000000 +4268.000000 +4242.000000 +4164.000000
     +1316.000000 +1336.000000 +1316.000000 +1256.000000

     Matches Matlab:

     >> ifft2(fft2([1:4;(5:8)*2;(9:12)*3]).*fft2([1:4;(5:8)*2;(9:12)*3]))

     ans =

     1.0e+03 *

     3.3140    3.3400    3.3140    3.2360
     4.2420    4.2680    4.2420    4.1640
     1.3160    1.3360    1.3160    1.2560
  */


  puts("");
  puts("Test real, Type-II symmetric psf_2d:");
  puts("------------------------------------");

  rs22_psf_2d_printf(p_rs22);

  rs22_psf_2d_dft(p_rs22);

  printf("\n");
  rs22_psf_2d_printf(p_rs22);

  printf("\n");
  rs22_psf_2d_printf_dft(p_rs22);

  /* Result:

     (+120.000000 +0.000000) (-2.000000 -0.828427) (-8.000000 -8.000000) (-2.000000 -4.828427) (+0.000000 +0.000000) (-2.000000 +4.828427) (-8.000000 +8.000000) (-2.000000 +0.828427)
     (+0.000000 +0.000000) (-11.021178 -14.363081) (+0.000000 +0.000000) (-0.978820 +7.434877) (+0.000000 +0.000000) (+5.949383 -4.565122) (+0.000000 +0.000000) (-17.949383 -2.363081)
     (+0.000000 +0.000001) (-0.565122 -4.292529) (-2.928203 +10.928202) (+6.363081 -8.292527) (+0.000000 +0.000000) (-10.363079 +1.364325) (+10.928202 +2.928203) (-3.434878 -2.635674)
     (+0.000000 +0.000000) (+0.000000 +0.000000) (+0.000000 +0.000000) (+0.000000 +0.000000) (+0.000000 +0.000000) (+0.000000 +0.000000) (+0.000000 +0.000000) (+0.000000 +0.000000)
     (+0.000000 -0.000001) (-3.434878 +2.635674) (+10.928202 -2.928202) (-10.363080 -1.364326) (+0.000000 +0.000000) (+6.363081 +8.292528) (-2.928202 -10.928202) (-0.565122 +4.292529)
     (+0.000000 +0.000000) (-17.949383 +2.363082) (+0.000000 +0.000000) (+5.949383 +4.565122) (+0.000000 +0.000000) (-0.978821 -7.434878) (+0.000000 +0.000000) (-11.021177 +14.363083)


     Matches Matlab:

     >> A = A = [1:4; [2 3 4 1]; [3 4 1 2]];
     >> B = [A, fliplr(A); flipud(A), fliplr(flipud(A))];
     >> fft2(B)

     ans =

        1.0e+02 *

     Columns 1 through 3

     1.2000            -0.0200 - 0.0083i  -0.0800 - 0.0800i
          0            -0.1102 - 0.1436i        0
          0            -0.0057 - 0.0429i  -0.0293 + 0.1093i
          0                  0                  0
          0            -0.0343 + 0.0264i   0.1093 - 0.0293i
          0            -0.1795 + 0.0236i        0

     Columns 4 through 6

     -0.0200 - 0.0483i        0            -0.0200 + 0.0483i
     -0.0098 + 0.0743i        0             0.0595 - 0.0457i
      0.0636 - 0.0829i        0            -0.1036 + 0.0136i
           0                  0                  0
     -0.1036 - 0.0136i        0             0.0636 + 0.0829i
      0.0595 + 0.0457i        0            -0.0098 - 0.0743i

     Columns 7 through 8

     -0.0800 + 0.0800i  -0.0200 + 0.0083i
           0            -0.1795 - 0.0236i
      0.1093 + 0.0293i  -0.0343 - 0.0264i
           0                  0
     -0.0293 - 0.1093i  -0.0057 + 0.0429i
           0            -0.1102 + 0.1436i

  */


  puts("");
  puts("Test filtering with a real, Type-II symmetric impulse response (real signal):");
  puts("-----------------------------------------------------------------------------");

  nx = 4;
  ny = 4;

  r_psf_2d_destroy(&p_r);
  rs22_psf_2d_destroy(&p_rs22);

  p_r = r_psf_2d_create(nx, ny);
  p_rs22 = rs22_psf_2d_create((int) nx/2, (int) ny/2);


  for (i = 0; i < p_r->ny_phy; i++) {
    for (j = 0; j < p_r->nx_phy; j++) {
      p_r->h[i + j*p_r->ny_phy]  = j + i*nx + 1;
    }
  }

  for (i = 0; i < p_rs22->ny_phy; i++) {
    for (j = 0; j < p_rs22->nx_phy; j++) {
      p_rs22->h[i + j*p_rs22->ny_phy]  = j + i*p_rs22->nx_phy + 1 + nx*ny;
    }
  }

  r_psf_2d_printf(p_r);
  printf("\n");

  rs22_psf_2d_printf(p_rs22);
  printf("\n");

  rs22_psf_2d_dft(p_rs22);
  r_psf_2d_dft(p_r);

  rs22_psf_2d_execute_r(p_rs22, p_r);

  r_psf_2d_idft(p_r);
  r_psf_2d_printf(p_r);
  printf("\n");

  /* Result:

     +2588.000000 +2580.000000 +2572.000000 +2580.000000
     +2524.000000 +2516.000000 +2508.000000 +2516.000000
     +2460.000000 +2452.000000 +2444.000000 +2452.000000
     +2524.000000 +2516.000000 +2508.000000 +2516.000000

     Matches Matlab:

     >> A = [1:4; 5:8; 9:12; 13:16];
     >> B = [17 18 18 17; 19 20 20 19; 19 20 20 19; 17 18 18 17];
     >> ifft2(fft2(A).*fft2(B))

     ans =

        2588        2580        2572        2580
        2524        2516        2508        2516
        2460        2452        2444        2452
        2524        2516        2508        2516

  */


  c_psf_2d_destroy(&p_c);
  r_psf_2d_destroy(&p_r);
  r_psf_2d_destroy(&p_r2);
  rs22_psf_2d_destroy(&p_rs22);

  fftwe_cleanup();

  return 0;
}
