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
  sparse_rcs *Q_sqrt = NULL;
  rs22_psf *h = NULL;
  
  int res = 8;
  int N = res*res;
  int TL = 4*1024;
  int K_phy = 2;
  int K_log;
  int K;
  int offset;
  ensemble *e = NULL;
  vector *u = NULL;
  r_psf *n = NULL;
  
  multi_sw *sw = NULL;
  int l;


  RANDN_DEBUG = True;
  
  randn_init();

  Q_sqrt = sparse_rcs_import("/tmp/Q_sqrt_test_2D");

  K_log = K_phy * 2;
  K = res + K_log - 1;
  if (is_odd(K)) {
    K++;
  }
  h = rs22_psf_import_zp("/tmp/h_psf_test", K_phy, K_phy, K/2, K/2);
  assert(h->ny_log == h->nx_log);
  assert(h->ny_log == K);
  offset = K_phy - 1;
  
  n = r_psf_create(K, K);

  e = ensemble_create(N, TL);
  ensemble_set0(e);

  u = vector_create(N);

  sw = multi_sw_create(2);

  assert(Q_sqrt->m == N);
  assert(Q_sqrt->n == K*K);
  
  multi_sw_start(sw, 0);
  for (l = 0; l < e->L; l++) {
    randn_v_rect(u, Q_sqrt);
    ecopy(u->n, u->v, 1, &(e->X->v[0][l]), e->L);
  }
  multi_sw_stop(sw, 0);
  
  full_r_export("/tmp/e_convmtx", e->X);


  randn_reset();
  
  rs22_psf_dft(h);
  ensemble_set0(e);
  
  multi_sw_start(sw, 1);
  for (l = 0; l < e->L; l++) {
    randn_v_conv_2D_add(&(e->X->v[0][l]), e->L, n, h, res, res, offset, offset);
  }
  multi_sw_stop(sw, 1);

  full_r_export("/tmp/e_conv", e->X);

  
  ensemble_set0(e);
  randn_reset();
  
  for (l = 0; l < e->N * e->L; l++) {
    e->X->v_vector[l] = randn(1);
  }

  full_r_export("/tmp/e_unit", e->X);
  
  multi_sw_printf(sw);

  sparse_rcs_destroy(&Q_sqrt);
  rs22_psf_destroy(&h);
  ensemble_destroy(&e);
  vector_destroy(&u);
  r_psf_destroy(&n);
  multi_sw_destroy(&sw);
  
  randn_exit();

  fftwe_cleanup();
  
  return 0;
}
