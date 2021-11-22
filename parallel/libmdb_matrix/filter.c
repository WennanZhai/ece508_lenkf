#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#include "filter.h"
#include "elem.h"
#include "eblas.h"
#include "util.h"


/***********
 * c_filter
 ***********/

/* An impulse response filter. */

c_filter *c_filter_create(int n) {
  c_filter *v;

  assert(n > 0);

  v = malloc(sizeof(c_filter));
  assert(v);

  v->h = fftwe_malloc(sizeof(fftwe_complex) * n);
  assert(v->h);

  v->n = n;
  v->d = S;

  /* On my machine, I have seen only extremely marginal gains by using
     a MEASURE based plan and opposed to an ESTIMATE plan */

  v->forward_plan = fftwe_plan_dft_1d(n, v->h, v->h,
				      FFTW_FORWARD, FFTW_ESTIMATE);

  v->backward_plan = fftwe_plan_dft_1d(n, v->h, v->h,
				      FFTW_BACKWARD, FFTW_ESTIMATE);

  return v;
}

void c_filter_destroy(c_filter **v) {
  assert(v);
  assert(*v);

  fftwe_free((*v)->h);
  fftwe_destroy_plan((*v)->forward_plan);
  fftwe_destroy_plan((*v)->backward_plan);

  free(*v);
  *v = NULL;
}

void c_filter_dft(c_filter *v) {
  assert(v);

  fftwe_execute(v->forward_plan);
  v->d = fftwe_domain_flip(v->d);
}

void c_filter_idft(c_filter *v) {
  assert(v);
  assert(v->d == F);

  /* NOTE:
   *
   * The ifft can also be computed in the following way:
   *    ifft(H) = fft(conj(H))/N
   *
   * Conjugatation is an O(N) operation:
   *
   *    escal((int) v->n, -1, &(v->h[0][1]), 2);
   *
   * Currently, I compute a backward plan.  Under the assumption that
   * the storage and computation of the backward plan is cheap, it is
   * better to just have a backward plan ready to do the inverse
   * transform.
   *
   */

  fftwe_execute(v->backward_plan);
  escal(2*v->n, ((elem) 1)/v->n, (elem *) v->h, 1);

  v->d = S;
}

void c_filter_printf(c_filter *v) {
  int i;

  assert(v);

  for (i = 0; i < v->n; i++) {
    if (i == v->n - 1) {
      fftwe_complex_printf((const fftwe_complex *) v->h[i]);
    }
    else {
      fftwe_complex_printf_s((const fftwe_complex *) v->h[i]);
    }
  }
}

void c_filter_set_real0(c_filter *v) {
  assert(v);

  escal(v->n, 0, &(v->h[0][0]), 2);
}

void c_filter_set_imag0(c_filter *v) {
  assert(v);

  escal(v->n, 0, &(v->h[0][1]), 2);
}

void c_filter_set0(c_filter *v) {
  assert(v);

  escal(2*v->n, 0, (elem *) v->h, 1);
}



/******************************************************************************/

/***********
 * r_filter
 ***********/


r_filter *r_filter_create(int n) {
  r_filter *v;

  assert(n > 0);

  v = malloc(sizeof(r_filter));
  assert(v);

  v->n_log = n;
  v->n_phy = ((floor(n/2)) + 1) * 2;

  /* See note on p. 6 of the FFTW manual 3.1 for the size of the malloc */
  v->h = fftwe_malloc(sizeof(elem) * v->n_phy);
  assert(v->h);

  v->d = S;

  v->forward_plan = fftwe_plan_dft_r2c_1d(n, v->h, (fftwe_complex *) v->h,
					  FFTW_ESTIMATE);

  v->backward_plan = fftwe_plan_dft_c2r_1d(n, (fftwe_complex *) v->h, v->h,
					   FFTW_ESTIMATE);

  return v;
}

void r_filter_destroy(r_filter **v) {
  assert(v);
  assert(*v);

  fftwe_free((*v)->h);

  fftwe_destroy_plan((*v)->forward_plan);
  fftwe_destroy_plan((*v)->backward_plan);

  free(*v);
  *v = NULL;
}

void r_filter_dft(r_filter *v) {
  assert(v);
  assert(v->d == S);

  fftwe_execute(v->forward_plan);
  v->d = fftwe_domain_flip(v->d);
}

void r_filter_idft(r_filter *v) {
  assert(v);
  assert(v->d == F);

  fftwe_execute(v->backward_plan);
  escal(v->n_log, ((elem) 1)/v->n_log, v->h, 1);

  v->d = fftwe_domain_flip(v->d);
}

void r_filter_printf(r_filter *v) {
  int i;

  assert(v);

  if (v->d == S) {
    for (i = 0; i < v->n_log; i++) {
      printf_elem_s(v->h[i]);
    }
  }
  else if(v->d == F) {
    for (i = 0; i < floor(v->n_log/2) + 1; i++) {
      printf_z_elem_s((const fftwe_complex *) &(v->h[2*i]));
    }
  }
  else {
    assert(0);
  }
}

void r_filter_set0(r_filter *v) {
  assert(v);

  set0(&(v->h[0]), v->n_phy);
}

/* The dft of a real signal is Hermitian symmetric. */
void r_filter_dft_get(const r_filter *v, int k, z_elem *H_k) {
  int k_half;
  z_elem *X_k;

  assert(v);
  assert(H_k);
  assert(v->d == F);
  assert(k >= 0 && k < v->n_log);

  k_half = floor(v->n_log/2);

  if (k < k_half + 1) {
    X_k = (z_elem *) &(v->h[2*k]);
    (*H_k)[0] = (*X_k)[0];
    (*H_k)[1] = (*X_k)[1];
  }
  else {
    X_k = (z_elem *) &(v->h[2*(v->n_log - k)]);
    (*H_k)[0] =  (*X_k)[0];
    (*H_k)[1] = -(*X_k)[1];
  }
}



/******************************************************************************/

/*************
 * rs1_filter
 *************/


rs1_filter *rs1_filter_create(int n_phy) {
  rs1_filter *v;

  assert(n_phy > 0);

  v = malloc(sizeof(rs1_filter));
  assert(v);

  v->h = fftwe_malloc(sizeof(elem) * n_phy);
  assert(v->h);

  v->n_phy = n_phy;
  v->n_log = 2*(n_phy-1);
  v->d = S;

  v->plan = fftwe_plan_r2r_1d(n_phy, v->h, v->h,
  			      FFTW_REDFT00, FFTW_ESTIMATE);

  return v;
}

rs1_filter *rs1_filter_import(char *filename) {
  rs1_filter *v;
  FILE *fid;
  int n_phy;
  int r;

  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  r = fread(&n_phy, sizeof(int), 1, fid);
  assert(r == 1);

  v = rs1_filter_create(n_phy);

  r = fread(v->h, sizeof(elem), n_phy, fid);
  assert(r == n_phy);

  fclose(fid);

  return v;
}


rs1_filter *rs1_filter_import_zp(char *filename, int n_phy_zp) {
  rs1_filter *v;
  FILE *fid;
  int n_phy;
  int r;

  assert(filename);
  assert(n_phy_zp > 0);

  fid = fopen(filename, "r");
  assert(fid);

  r = fread(&n_phy, sizeof(int), 1, fid);
  assert(r == 1);
  assert(n_phy_zp >= n_phy);

  v = rs1_filter_create(n_phy_zp);
  set0(v->h, n_phy_zp);

  r = fread(v->h, sizeof(elem), n_phy, fid);
  assert(r == n_phy);

  fclose(fid);

  return v;
}

void rs1_filter_destroy(rs1_filter **v) {
  assert(v);
  assert(*v);

  fftwe_free((*v)->h);
  fftwe_destroy_plan((*v)->plan);

  free(*v);
  *v = NULL;
}

void rs1_filter_dft(rs1_filter *v) {
  assert(v);

  fftwe_execute(v->plan);
  v->d = fftwe_domain_flip(v->d);
}

void rs1_filter_idft(rs1_filter *v) {
  assert(v);
  assert(v->d == F);

  fftwe_execute(v->plan);

  /* See (8.165), O&S2 */
  escal(v->n_phy, ((elem) 1)/(2*v->n_phy - 2), (elem *) v->h, 1);

  v->d = S;
}

void rs1_filter_printf(const rs1_filter *v) {
  int i;

  assert(v);

  for (i = 0; i < v->n_phy; i++) {
    printf_elem_s(v->h[i]);
  }
}

void rs1_filter_set0(rs1_filter *v) {
  assert(v);

  set0(v->h, v->n_phy);
}

elem rs1_filter_dft_get(const rs1_filter *v, int k) {
  assert(v);
  assert(k >= 0 && k < v->n_log);
  assert(v->d == F);

  if (k < v->n_phy) {
    return v->h[k];
  }
  else {
    return v->h[2*v->n_phy - 2 - k];
  }
}


/* Note:

   There seems to be no speedup of the BLAS version of the filter code
   and the looped version given below.
*/

void rs1_filter_execute(const rs1_filter *v, c_filter *x) {
  assert(v);
  assert(x);
  assert(v->n_log == x->n);
  assert(v->d == F);
  assert(x->d == F);


  /* Element by element multiply real and imaginary parts by the
     impulse response v->h. */

  /* Case: 0 <= k < v->n_phy

     X[k] = X[k] * H[k]
  */

  etbmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
	v->n_phy, 0, v->h, 1, &(x->h[0][0]), 2);

  etbmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
	v->n_phy, 0, v->h, 1, &(x->h[0][1]), 2);


  /* Case: v->n_phy <= k < v->n_log

     X[k] = X[k] * H[V->n_log - 1 - k]
  */

  if (v->n_phy > 2) {
    etbmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
	  v->n_phy - 2, 0, &(v->h[1]), 1, &(x->h[v->n_phy][0]), -2);

    etbmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
	  v->n_phy - 2, 0, &(v->h[1]), 1, &(x->h[v->n_phy][1]), -2);
  }
}

void rs1_filter_execute_noblas(const rs1_filter *v, c_filter *x) {
  int k;
  elem H_k;

  assert(v);
  assert(x);
  assert(v->n_log == x->n);
  assert(v->d == F);
  assert(x->d == F);


  for (k = 0; k < x->n; k++) {
    H_k = rs1_filter_dft_get(v, k);
    x->h[k][0] *= H_k;
    x->h[k][1] *= H_k;
  }
}


/******************************************************************************/

/*************
 * rs2_filter
 *************/

rs2_filter *rs2_filter_create(int n_phy) {
  rs2_filter *v;

  assert(n_phy > 0);

  v = malloc(sizeof(rs2_filter));
  assert(v);

  v->h = fftwe_malloc(sizeof(elem) * n_phy);
  assert(v->h);

  v->n_phy = n_phy;
  v->n_log = 2*n_phy;
  v->d = S;

  v->forward_plan = fftwe_plan_r2r_1d(n_phy, v->h, v->h,
				      FFTW_REDFT10, FFTW_ESTIMATE);

  v->backward_plan = fftwe_plan_r2r_1d(n_phy, v->h, v->h,
				       FFTW_REDFT01, FFTW_ESTIMATE);

  return v;
}

rs2_filter *rs2_filter_import(char *filename) {
  rs2_filter *v;
  FILE *fid;
  int n_phy;
  int r;

  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  r = fread(&n_phy, sizeof(int), 1, fid);
  assert(r == 1);

  v = rs2_filter_create(n_phy);

  r = fread(v->h, sizeof(elem), n_phy, fid);
  assert(r == n_phy);

  fclose(fid);

  return v;
}

rs2_filter *rs2_filter_import_zp(const char *filename, int n_phy,
				 int n_phy_zp) {
  rs2_filter *v;
  FILE *fid;
  int n_phy_fread;
  int r;

  assert(filename);
  assert(n_phy > 0);
  assert(n_phy_zp > 0 && n_phy_zp >= n_phy);

  fid = fopen(filename, "r");
  assert(fid);

  r = fread(&n_phy_fread, sizeof(int), 1, fid);
  assert(r == 1);
  assert(n_phy == n_phy_fread);

  v = rs2_filter_create(n_phy_zp);
  set0(v->h, n_phy_zp);

  r = fread(v->h, sizeof(elem), n_phy, fid);
  assert(r == n_phy);

  fclose(fid);

  return v;
}

void rs2_filter_destroy(rs2_filter **v) {
  assert(v);
  assert(*v);

  fftwe_free((*v)->h);
  fftwe_destroy_plan((*v)->forward_plan);
  fftwe_destroy_plan((*v)->backward_plan);

  free(*v);
  *v = NULL;
}

void rs2_filter_dft(rs2_filter *v) {
  assert(v);

  fftwe_execute(v->forward_plan);
  v->d = fftwe_domain_flip(v->d);
}

void rs2_filter_idft(rs2_filter *v) {
  assert(v);
  assert(v->d == F);

  fftwe_execute(v->backward_plan);

  /* See (8.175), O&S2 */
  escal(v->n_phy, ((elem) 1)/(2*v->n_phy), (elem *) v->h, 1);

  v->d = S;
}

void rs2_filter_printf(const rs2_filter *v) {
  int i;

  assert(v);

  for (i = 0; i < v->n_phy; i++) {
    printf_elem_s(v->h[i]);
  }
}

void rs2_filter_printf_dft(const rs2_filter *v) {
  int k;
  z_elem H_k;

  assert(v);
  assert(v->d == F);

  for (k = 0; k < v->n_log; k++) {
    rs2_filter_dft_get(v, k, &H_k);
    printf_z_elem_s((const z_elem *) &H_k);
  }
}

void rs2_filter_set0(rs2_filter *v) {
  assert(v);

  set0(v->h, v->n_phy);
}

void rs2_filter_dft_get(const rs2_filter *v, int k, z_elem *H_k) {
  z_elem z;
  elem theta;

  assert(v);
  assert(k >= 0 && k < v->n_log);
  assert(v->d == F);

  if (k == 0) {
    (*H_k)[0] = v->h[0];
    (*H_k)[1] = 0;
  }
  else if (k <= v->n_phy - 1) {
    theta = M_PI * k / (2 * v->n_phy);
    expj(theta, &z);

    (*H_k)[0] = v->h[k] * z[0];
    (*H_k)[1] = v->h[k] * z[1];
  }
  else if (k == v->n_phy) {
    (*H_k)[0] = 0;
    (*H_k)[1] = 0;
  }
  else {
    theta = M_PI * k / (2 * v->n_phy);
    expj(theta, &z);

    (*H_k)[0] = v->h[2*v->n_phy - k] * -z[0];
    (*H_k)[1] = v->h[2*v->n_phy - k] * -z[1];
  }
}

void rs2_filter_execute_c(const rs2_filter *v, c_filter *x) {
  int k;
  z_elem H_k;

  assert(v);
  assert(x);
  assert(v->n_log == x->n);
  assert(v->d == F);
  assert(x->d == F);


  for (k = 0; k < x->n; k++) {
    rs2_filter_dft_get(v, k, &H_k);
    zmul(&(x->h[k]), (const z_elem *) &H_k);
  }
}

void rs2_filter_execute_r(const rs2_filter *v, r_filter *x) {
  z_elem H_k;
  z_elem *X_k;
  int k;

  assert(v);
  assert(x);
  assert(v->n_log == x->n_log);
  /* Hence, x and v are both even logical length. */
  assert(v->d == F);
  assert(x->d == F);


  for (k = 0; k < (int) x->n_log/2 + 1; k++) {
    rs2_filter_dft_get(v, k, &H_k);
    X_k = (z_elem *) &(x->h[2*k]);
    zmul(X_k, (const z_elem *) &H_k);
  }
}
