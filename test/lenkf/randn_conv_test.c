#include <assert.h>

#ifdef LENKF_FLOAT_ELEM
#include "mdb_matrix_s.h"
#elif defined LENKF_DOUBLE_ELEM
#include "mdb_matrix_d.h"
#else
#error ?
#endif

#include "ensemble.h"
#include "randn.h"
#include "util.h"


extern boolean RANDN_DEBUG;

int main(void) {
  sparse_rcs *Q_sqrt;
  rs2_filter *h;
  
  int N = 32;
  int TL = 1024;
  int K_phy = 2;
  int K_log;
  int K;
  int offset;
  ensemble *e;
  vector *u;
  r_filter *n;
  
  multi_sw *sw;
  int l;


  RANDN_DEBUG = True;
  
  randn_init();

  Q_sqrt = sparse_rcs_import("/tmp/Q_sqrt_test");

  K_log = K_phy * 2;
  K = N + K_log - 1;
  if (is_odd(K)) {
    K++;
  }
  h = rs2_filter_import_zp("/tmp/h_test", K_phy, K/2);
  assert(K == h->n_log);
  offset = K_phy - 1;
  
  n = r_filter_create(K);
  
  e = ensemble_create(N, TL);
  ensemble_set0(e);

  u = vector_create(N);

  sw = multi_sw_create(2);
  
  assert(Q_sqrt->m == Q_sqrt->n);
  assert(Q_sqrt->n == N);

  multi_sw_start(sw, 0);
  for (l = 0; l < e->L; l++) {
    randn_v(u, Q_sqrt);
    eaxpy(u->n, 1, u->v, 1, &(e->X->v[0][l]), e->L);
  }
  multi_sw_stop(sw, 0);

  full_r_export("/tmp/e_chol", e->X);

  ensemble_set0(e);

  rs2_filter_dft(h);
  
  multi_sw_start(sw, 1);
  for (l = 0; l < e->L; l++) {
    randn_v_conv_1D(u, n, h, offset);
    eaxpy(u->n, 1, u->v, 1, &(e->X->v[0][l]), e->L);
  }
  multi_sw_stop(sw, 1);

  full_r_export("/tmp/e_conv", e->X);
  
  multi_sw_printf(sw);
  
  sparse_rcs_destroy(&Q_sqrt);
  rs2_filter_destroy(&h);
  ensemble_destroy(&e);
  vector_destroy(&u);
  r_filter_destroy(&n);
  multi_sw_destroy(&sw);
  
  randn_exit();

  fftwe_cleanup();
  
  return 0;
}
