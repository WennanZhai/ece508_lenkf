#include <stdio.h>
#include <assert.h>

#ifdef FLOAT_ELEM_TEST
#include "mdb_matrix_s.h"
#elif defined DOUBLE_ELEM_TEST
#include "mdb_matrix_d.h"
#else
#error 0
#endif


int main(int argc, char **argv) {
  int nx = 4;
  int ny = 3;
  int nz = 2;

  int i;

  c_psf_3d *A;

  A = c_psf_3d_create(nx, ny, nz);
  c_psf_3d_set_imag0(A);

  for (i = 0; i < nx * ny * nz; i++) {
#ifdef VECLIB
      A->h[i][0] = i+1;
#elif defined OPENBLAS
      A->h[i] = i+1;
#else
#error 0
#endif
  }

  puts("Test complex psf_3d:");
  puts("--------------------");

  puts("");
  c_psf_3d_printf(A);

  c_psf_3d_dft(A);
  puts("");
  c_psf_3d_printf(A);


  /* Result:


     Matches Matlab:

     A = reshape(reshape(1:24,2,4*3)', 3, 4, 2);

     >> fftn(A)

     ans(:,:,1) =

     1.0e+02 *

     3.0000            -0.7200 + 0.7200i  -0.7200            -0.7200 - 0.7200i
    -0.2400 + 0.1386i        0                  0                  0
    -0.2400 - 0.1386i        0                  0                  0


    ans(:,:,2) =

    -12     0     0     0
      0     0     0     0
      0     0     0     0

   */

  c_psf_3d_destroy(&A);

  fftwe_cleanup();

  return 0;
}
