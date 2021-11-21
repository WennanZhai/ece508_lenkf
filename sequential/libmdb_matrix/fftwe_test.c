#include <assert.h>

#ifdef FLOAT_ELEM_TEST
#include "mdb_matrix_s.h"
#elif defined DOUBLE_ELEM_TEST
#include "mdb_matrix_d.h"
#else
#error 0
#endif


int main(int argc, char **argv) {
  fftwe_plan C1;
  elem *h1;
  fftwe_plan C2;
  fftwe_plan C3;
  elem *h2;

  const int N = 4;
  int i;


  h1 = fftwe_malloc(N * sizeof(elem));
  assert(h1);

  C1 = fftwe_plan_r2r_1d(N, h1, h1, FFTW_REDFT00, FFTW_ESTIMATE);

  for (i = 0; i < N; i++) {
    h1[i] = N-i;
  }

  fftwe_execute(C1);

  printf("dct1(h1):\n");
  for (i = 0; i < N; i++) {
    printf_elem_s(h1[i]);
  }
  printf("\n");

  /* Corresponds to fft([[N:-1:1],[2:N-1]]) */
  /* np.fft.fft(list(range(N, 1, -1)) + list(range(1, N))) */

  fftwe_execute(C1);

  printf("\n");
  printf("idct1(dct1(h1)):\n");
  for (i = 0; i < N; i++) {
    printf_elem_s(h1[i] / (2*(N-1)));
  }
  printf("\n");

  fftwe_destroy_plan(C1);
  fftwe_free(h1);


  h2 = fftwe_malloc(N * sizeof(elem));
  assert(h2);

  C2 = fftwe_plan_r2r_1d(N, h2, h2, FFTW_REDFT10, FFTW_ESTIMATE);
  C3 = fftwe_plan_r2r_1d(N, h2, h2, FFTW_REDFT01, FFTW_ESTIMATE);

  for (i = 0; i < N; i++) {
    h2[i] = N-i;
  }

  fftwe_execute(C2);

  printf("\n");
  printf("dct2(h2):\n");
  for (i = 0; i < N; i++) {
    printf_elem_s(h2[i]);
  }
  printf("\n");

  /* Corresponds to fft([[N:-1:1],[1:N]]).*exp(-j*pi*(0:2*(N-1)+1)/(2*N)) */
  /* x = list(range(4, 0, -1)) + list(range(1, 5))
     np.fft.fft(x) * np.exp(-1j*np.pi*np.array(range(0, 2*N))/(2*N)) */

  fftwe_execute(C3);

  printf("\n");
  printf("idct2(dct2(h2)):\n");
  for (i = 0; i < N; i++) {
    printf_elem_s(h2[i] / (2*N));
  }
  printf("\n");

  /* Corresponds to list(range(N, 0, -1)) */

  fftwe_destroy_plan(C2);
  fftwe_destroy_plan(C3);
  fftwe_free(h2);


  fftwe_cleanup();

  return 0;
}
