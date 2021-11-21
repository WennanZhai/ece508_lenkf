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
  c_psf *p_c;
  r_psf *p_r;
  r_psf *p_r2;
  rs22_psf *p_rs22;
  rs22_psf *p_rs22_zp;

  int ny = 3;
  int nx = 4;

  fftwe_complex *c_ptr;
  elem *e_ptr, *e_ptr2;
  z_elem H_ky_kx;

  int i, j, count;


  p_c = c_psf_create(ny, nx);

  c_ptr = p_c->h;
  count = 1;
  for (i = 0; i < ny; i++) {
    for (j = 0; j < nx; j++) {
#ifdef VECLIB
      *c_ptr[0] = count*(i+1);
#elif defined OPENBLAS
      *c_ptr = count*(i+1);
#else
#error 0
#endif
      c_ptr++;
      count++;
    }
  }
  c_psf_set_imag0(p_c);

  p_r = r_psf_create(ny, nx);
  r_psf_set0(p_r);

  p_r2 = r_psf_create(ny, nx);
  r_psf_set0(p_r2);

  count = 1;
  for (i = 0; i < ny; i++) {
    e_ptr = &(p_r->h[i*p_r->nx_phy]);
    e_ptr2 = &(p_r2->h[i*p_r2->nx_phy]);
    for (j = 0; j < nx; j++) {
      *e_ptr = count*(i+1);
      *e_ptr2 = count*(i+1);
      e_ptr++;
      e_ptr2++;
      count++;
    }
  }

  p_rs22 = rs22_psf_create(ny, nx);

  e_ptr = p_rs22->h;
  for (i = 0; i < ny; i++) {
    for (j = 0; j < nx; j++) {
      *e_ptr = (i+j) % nx + 1;
      e_ptr++;
    }
  }


  puts("Test complex psf:");
  puts("-----------------");

  c_psf_printf(p_c);

  c_psf_dft(p_c);

  printf("\n");
  c_psf_printf(p_c);

  c_psf_idft(p_c);

  printf("\n");
  c_psf_printf(p_c);

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


  puts("");
  puts("Test real psf:");
  puts("--------------");

  r_psf_printf(p_r);

  r_psf_dft(p_r);

  printf("\n");
  r_psf_printf(p_r);

  printf("\n");
  for (i = 0; i < p_r->ny_log; i++) {
    for (j = 0; j < p_r->nx_log; j++) {
      r_psf_dft_get(p_r, i, j, &H_ky_kx);
      printf_z_elem_s((const z_elem *) H_ky_kx);
    }
    printf("\n");
  }
  printf("\n");

  r_psf_idft(p_r);

  printf("\n");
  r_psf_printf(p_r);

  printf("\n");
  r_psf_printf(p_r2);

  r_psf_dft(p_r);
  r_psf_dft(p_r2);
  r_psf_execute_r(p_r, p_r2);

  r_psf_idft(p_r2);
  printf("\n");
  r_psf_printf(p_r2);


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
  puts("Test real, Type-II symmetric psf:");
  puts("---------------------------------");

  rs22_psf_printf(p_rs22);

  rs22_psf_dft(p_rs22);

  printf("\n");
  rs22_psf_printf(p_rs22);

  printf("\n");
  for (i = 0; i < p_rs22->ny_log; i++) {
    for (j = 0; j < p_rs22->nx_log; j++) {
      rs22_psf_dft_get(p_rs22, i, j, &H_ky_kx);
      printf_z_elem_s((const z_elem *) H_ky_kx);
    }
    printf("\n");
  }

  /* Result:

     (+120.000000 +0.000000) (-1.999999 -0.828427) (-8.000000 -8.000000) (-2.000000 -4.828426) (+0.000000 +0.000000) (-2.000000 +4.828427) (-8.000000 +8.000000) (-1.999999 +0.828427)
     (+0.000000 +0.000000) (-11.021178 -14.363080) (+0.000000 +0.000000) (-0.978820 +7.434876) (+0.000000 +0.000000) (+5.949382 -4.565122) (+0.000000 +0.000000) (-17.949383 -2.363080)
     (+0.000000 +0.000000) (-0.565122 -4.292529) (-2.928204 +10.928203) (+6.363081 -8.292528) (+0.000000 +0.000000) (-10.363081 +1.364325) (+10.928203 +2.928204) (-3.434878 -2.635674)
     (+0.000000 +0.000000) (+0.000000 +0.000000) (+0.000000 +0.000000) (+0.000000 +0.000000) (+0.000000 +0.000000) (+0.000000 +0.000000) (+0.000000 +0.000000) (+0.000000 +0.000000)
     (+0.000000 +0.000000) (-3.434878 +2.635674) (+10.928203 -2.928202) (-10.363081 -1.364326) (+0.000000 +0.000000) (+6.363081 +8.292529) (-2.928202 -10.928203) (-0.565122 +4.292529)
     (+0.000000 +0.000000) (-17.949383 +2.363082) (+0.000000 +0.000000) (+5.949382 +4.565121) (+0.000000 +0.000000) (-0.978821 -7.434877)  (+0.000000 -0.000000) (-11.021177 +14.363083)


     Matches Matlab:

     >> A = [1:4; [2 3 4 1]; [3 4 1 2]];
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

  r_psf_destroy(&p_r);
  rs22_psf_destroy(&p_rs22);

  p_r = r_psf_create(ny, nx);
  p_rs22 = rs22_psf_create((int) ny/2, (int) nx/2);

  count = 1;
  for (i = 0; i < ny; i++) {
    e_ptr = &(p_r->h[i*p_r->nx_phy]);
    for (j = 0; j < nx; j++) {
      *e_ptr = count;
      count++;
      e_ptr++;
    }
  }

  e_ptr = p_rs22->h;
  for (i = 0; i < p_rs22->ny_phy; i++) {
    for (j = 0; j < p_rs22->nx_phy; j++) {
      *e_ptr = count;
      count++;
      e_ptr++;
    }
  }

  r_psf_printf(p_r);
  printf("\n");

  rs22_psf_printf(p_rs22);
  printf("\n");

  rs22_psf_dft(p_rs22);
  r_psf_dft(p_r);
  rs22_psf_execute_r(p_rs22, p_r);

  r_psf_idft(p_r);
  r_psf_printf(p_r);
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

  rs22_psf_idft(p_rs22);
  rs22_psf_export("/tmp/p_rs22_psf_test", p_rs22);

  p_rs22_zp = rs22_psf_import_zp("/tmp/p_rs22_psf_test",
				 (int) ny/2, (int) nx/2,
				 (int) ny/2 + 2, (int) nx/2 + 3);

  rs22_psf_printf(p_rs22_zp);

  c_psf_destroy(&p_c);
  r_psf_destroy(&p_r);
  r_psf_destroy(&p_r2);
  rs22_psf_destroy(&p_rs22);
  rs22_psf_destroy(&p_rs22_zp);

  fftwe_cleanup();

  return 0;
}
