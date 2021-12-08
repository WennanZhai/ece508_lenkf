#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "psf_2d.h"
#include "eblas.h"



/******************************************************************************/

/**********
 * c_psf_2d
 **********/

c_psf_2d *c_psf_2d_create(int nx, int ny) {
  c_psf_2d *p;

  assert(nx > 0);
  assert(ny > 0);

  p = malloc(sizeof(c_psf_2d));
  assert(p);

  p->h = fftwe_malloc(sizeof(fftwe_complex) * nx * ny);
  assert(p->h);

  p->nx = nx;
  p->ny = ny;
  p->N = nx*ny;
  p->d = S;

  p->forward_plan = fftwe_plan_dft_2d(nx, ny, p->h, p->h,
				      FFTW_FORWARD, FFTW_ESTIMATE);

  p->backward_plan = fftwe_plan_dft_2d(nx, ny, p->h, p->h,
				       FFTW_BACKWARD, FFTW_ESTIMATE);

  return p;
}

void c_psf_2d_destroy(c_psf_2d **p) {
  assert(p);
  assert(*p);

  fftwe_destroy_plan((*p)->forward_plan);
  fftwe_destroy_plan((*p)->backward_plan);

  free((*p)->h);

  free(*p);
  *p = NULL;
}

void c_psf_2d_dft(c_psf_2d *p) {
  assert(p);

  fftwe_execute(p->forward_plan);
  p->d = fftwe_domain_flip(p->d);
}

void c_psf_2d_idft(c_psf_2d *p) {
  assert(p);
  assert(p->d == F);

  fftwe_execute(p->backward_plan);
  escal(2*p->N, ((elem) 1)/p->N, (elem *) p->h, 1);
  p->d = fftwe_domain_flip(p->d);
}

void c_psf_2d_printf(c_psf_2d *p) {
  int ix, iy;

  assert(p);

  for (iy = 0; iy < p->ny; iy++) {
    for (ix = 0; ix < p->nx; ix++) {
      printf_z_elem_s((const z_elem *) &p->h[iy + ix*p->ny]);
    }
    printf("\n");
  }
}

void c_psf_2d_set_real0(c_psf_2d *p) {
  assert(p);

  escal(p->N, 0, &(p->h[0][0]), 2);
}

void c_psf_2d_set_imag0(c_psf_2d *p) {
  assert(p);

  escal(p->N, 0, &(p->h[0][1]), 2);
}

void c_psf_2d_set0(c_psf_2d *p) {
  assert(p);

  escal(2*p->N, 0, &(p->h[0][0]), 1);
}


/******************************************************************************/

/**********
 * r_psf_2d
 **********/

r_psf_2d *r_psf_2d_create(int nx, int ny) {
  r_psf_2d *p;

  assert(nx > 0);
  assert(ny > 0);

  p = malloc(sizeof(r_psf_2d));
  assert(p);

  p->nx_log = nx;
  p->ny_log = ny;
  p->N_log = nx*ny;

  /* See Section 2.4 of the FFTW manual 3.1 for a discussion of the
     necessary physical size of p->h to acoomodate an inplace
     transform. */
  p->nx_phy = nx;
  p->ny_phy = (floor(ny/2) + 1) * 2;

  p->N_phy = p->nx_phy * p->ny_phy;

  p->h = fftwe_malloc(sizeof(elem) * p->N_phy);
  assert(p->h);

  p->d = S;

  p->forward_plan = fftwe_plan_dft_r2c_2d(nx, ny, p->h, (fftwe_complex *) p->h,
					  FFTW_ESTIMATE);

  p->backward_plan = fftwe_plan_dft_c2r_2d(nx, ny, (fftwe_complex *) p->h, p->h,
					   FFTW_ESTIMATE);

  return p;
}

void r_psf_2d_destroy(r_psf_2d **p) {
  assert(p);
  assert(*p);

  fftwe_destroy_plan((*p)->forward_plan);
  fftwe_destroy_plan((*p)->backward_plan);

  free((*p)->h);

  free(*p);
  *p = NULL;
}

void r_psf_2d_dft(r_psf_2d *p) {
  assert(p);

  fftwe_execute(p->forward_plan);
  p->d = fftwe_domain_flip(p->d);
}

void r_psf_2d_idft(r_psf_2d *p) {
  assert(p);
  assert(p->d == F);

  fftwe_execute(p->backward_plan);
  escal(p->N_phy, ((elem) 1)/p->N_log, p->h, 1);

  p->d = fftwe_domain_flip(p->d);
}

void r_psf_2d_printf(r_psf_2d *p) {
  int ix, iy;

  assert(p);

  if (p->d == S) {
    for (iy = 0; iy < p->ny_log; iy++) {
      for (ix = 0; ix < p->nx_log; ix++) {
	printf_elem_s(p->h[iy + ix*p->ny_phy]);
      }
      printf("\n");
    }
  }
  else if (p->d == F) {
    for (iy = 0; iy < p->ny_phy/2; iy++) {
      for (ix = 0; ix < p->nx_log; ix++) {
	printf_z_elem_s((const z_elem *) &(p->h[2*(iy + ix*p->ny_phy/2)]));
      }
      printf("\n");
    }
  }
  else {
    assert(0);
  }
}

void r_psf_2d_printf_dft(r_psf_2d *p) {
  int k1, k2;
  z_elem H_kx_ky;

  assert(p);
  assert(p->d == F);

  for (k2 = 0; k2 < p->ny_log; k2++) {
    for (k1 = 0; k1 < p->nx_log; k1++) {
      r_psf_2d_dft_get(p, k1, k2, &H_kx_ky);
      printf_z_elem_s((const z_elem *) &H_kx_ky);
    }
    printf("\n");
  }
}

void r_psf_2d_set0(r_psf_2d *p) {
  assert(p);

  set0(p->h, p->N_phy);
}

void r_psf_2d_dft_get(const r_psf_2d *p, int kx, int ky, z_elem *H_kx_ky) {
  z_elem *X_k;
  int ny_log_half;
  int ny_phy_half;
  int k1, k2, index;

  assert(p);
  assert(H_kx_ky);
  assert(kx >= 0 && kx < p->nx_log);
  assert(ky >= 0 && ky < p->ny_log);
  assert(p->d == F);


  ny_log_half = floor(p->ny_log/2);
  ny_phy_half = floor(p->ny_phy/2);

  if (ky <= ny_log_half) {
    k1 = kx;
    k2 = ky;
  }
  else {
    k1 = abs(p->nx_log - kx) % p->nx_log;
    k2 = p->ny_log - ky;
  }

  index = 2*(k2 + k1*ny_phy_half);
  assert(index >= 0 && index < p->N_phy - 1);

  X_k = (z_elem *) &(p->h[index]);

  (*H_kx_ky)[0] = (*X_k)[0];

  if (ky <= ny_log_half) {
    (*H_kx_ky)[1] = (*X_k)[1];
  }
  else {
    (*H_kx_ky)[1] = -(*X_k)[1];
  }
}

void r_psf_2d_execute_r(const r_psf_2d *p_h, r_psf_2d *p_x) {
  int i;

  assert(p_h);
  assert(p_x);
  assert(p_h->nx_log == p_x->nx_log);
  assert(p_h->ny_log == p_x->ny_log);
  assert(p_h->d == F);
  assert(p_x->d == F);


  for (i = 0; i < p_h->N_phy; i += 2) {
    zmul((z_elem *) &(p_x->h[i]), (const z_elem *) &(p_h->h[i]));
  }
}


/******************************************************************************/

/*************
 * rs22_psf_2d
 *************/

rs22_psf_2d *rs22_psf_2d_create(int nx_phy, int ny_phy) {
  rs22_psf_2d *p;

  assert(nx_phy > 0);
  assert(ny_phy > 0);

  p = malloc(sizeof(rs22_psf_2d));
  assert(p);

  p->nx_phy = nx_phy;
  p->ny_phy = ny_phy;
  p->N_phy = nx_phy * ny_phy;

  p->nx_log = 2*nx_phy;
  p->ny_log = 2*ny_phy;
  p->N_log = 4 * nx_phy * ny_phy;

  p->h = fftwe_malloc(sizeof(elem) * nx_phy * ny_phy);
  assert(p->h);

  p->d = S;

  p->forward_plan = fftwe_plan_r2r_2d(nx_phy, ny_phy, p->h, p->h,
				      FFTW_REDFT10, FFTW_REDFT10,
				      FFTW_ESTIMATE);

  p->backward_plan = fftwe_plan_r2r_2d(nx_phy, ny_phy, p->h, p->h,
				       FFTW_REDFT01, FFTW_REDFT01,
				       FFTW_ESTIMATE);

  return p;
}

void rs22_psf_2d_destroy(rs22_psf_2d **p) {
  assert(p);
  assert(*p);

  fftwe_free((*p)->h);

  fftwe_destroy_plan((*p)->forward_plan);
  fftwe_destroy_plan((*p)->backward_plan);

  free(*p);
  *p = NULL;
}

void rs22_psf_2d_dft(rs22_psf_2d *p) {
  assert(p);

  fftwe_execute(p->forward_plan);
  p->d = fftwe_domain_flip(p->d);
}

void rs22_psf_2d_idft(rs22_psf_2d *p) {
  assert(p);
  assert(p->d == F);

  fftwe_execute(p->backward_plan);
  escal(p->N_phy, ((elem) 1) / p->N_log, p->h, 1);

  p->d = fftwe_domain_flip(p->d);
}

void rs22_psf_2d_printf(rs22_psf_2d *p) {
  int i, j;

  assert(p);

  for (i = 0; i < p->ny_phy; i++) {
    for (j = 0; j < p->nx_phy; j++) {
      printf_elem_s(p->h[i + j*p->ny_phy]);
    }
    printf("\n");
  }
}

void rs22_psf_2d_printf_dft(rs22_psf_2d *p) {
  int kx, ky;
  z_elem H_kx_ky;

  assert(p);
  assert(p->d == F);

  for (ky = 0; ky < p->ny_log; ky++) {
    for (kx = 0; kx < p->nx_log; kx++) {
      rs22_psf_2d_dft_get(p, kx, ky, &H_kx_ky);
      printf_z_elem_s((const z_elem *) &H_kx_ky);
    }
    printf("\n");
  }
}

void rs22_psf_2d_dft_get(const rs22_psf_2d *p, int kx, int ky,
			 z_elem *H_kx_ky) {
  z_elem z;
  elem theta;
  int k1, k2;
  z_elem H_kx, H_ky;

  assert(p);
  assert(H_kx_ky);
  assert(kx >= 0 && kx < p->nx_log);
  assert(ky >= 0 && ky < p->ny_log);


  if (kx == 0) {
    k1 = 0;
    H_kx[0] = 1;
    H_kx[1] = 0;
  }
  else if (kx <= p->nx_phy - 1) {
    theta = M_PI * kx / (2 * p->nx_phy);
    expj(theta, &z);

    k1 = kx;
    H_kx[0] = z[0];
    H_kx[1] = z[1];
  }
  else if (kx == p->nx_phy) {
    (*H_kx_ky)[0] = 0;
    (*H_kx_ky)[1] = 0;
    return;
  }
  else {
    theta = M_PI * kx / (2 * p->nx_phy);
    expj(theta, &z);

    k1 = 2*p->nx_phy - kx;
    H_kx[0] = -z[0];
    H_kx[1] = -z[1];
  }

  if (ky == 0) {
    k2 = 0;
    H_ky[0] = p->h[k2 + k1 * p->ny_phy];
    H_ky[1] = 0;
  }
  else if (ky <= p->ny_phy - 1) {
    theta = M_PI * ky / (2 * p->ny_phy);
    expj(theta, &z);

    k2 = ky;
    H_ky[0] = p->h[k2 + k1 * p->ny_phy] * z[0];
    H_ky[1] = p->h[k2 + k1 * p->ny_phy] * z[1];
  }
  else if (ky == p->ny_phy) {
    (*H_kx_ky)[0] = 0;
    (*H_kx_ky)[1] = 0;
    return;
  }
  else {
    theta = M_PI * ky / (2 * p->ny_phy);
    expj(theta, &z);

    k2 = 2*p->ny_phy - ky;
    H_ky[0] = p->h[k2 + k1 * p->ny_phy] * -z[0];
    H_ky[1] = p->h[k2 + k1 * p->ny_phy] * -z[1];
  }

  zmul(&H_ky, (const z_elem *) &H_kx);
  (*H_kx_ky)[0] = H_ky[0];
  (*H_kx_ky)[1] = H_ky[1];
}

rs22_psf_2d *rs22_psf_2d_import(char *filename) {
  FILE *fid;
  int nx_phy, ny_phy;
  rs22_psf_2d *p;
  int r;

  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  r = fread(&nx_phy, sizeof(int), 1, fid);
  assert(r == 1);

  r = fread(&ny_phy, sizeof(int), 1, fid);
  assert(r == 1);

  p = rs22_psf_2d_create(nx_phy, ny_phy);

  r = fread(p->h, sizeof(elem), p->N_phy, fid);
  assert(r == p->N_phy);

  fclose(fid);

  return p;
}

rs22_psf_2d *rs22_psf_2d_import_zp(const char *filename, int nx_phy, int ny_phy,
				   int nx_phy_zp, int ny_phy_zp) {
  FILE *fid;
  int nx_phy_fread, ny_phy_fread;
  rs22_psf_2d *p;
  int j, index;
  int r;

  assert(filename);
  assert(nx_phy > 0);
  assert(ny_phy > 0);
  assert(nx_phy_zp > 0 && nx_phy_zp >= nx_phy);
  assert(ny_phy_zp > 0 && ny_phy_zp >= ny_phy);

  fid = fopen(filename, "r");
  assert(fid);

  r = fread(&nx_phy_fread, sizeof(int), 1, fid);
  assert(r == 1);

  r = fread(&ny_phy_fread, sizeof(int), 1, fid);
  assert(r == 1);

  assert(nx_phy_fread == nx_phy);
  assert(ny_phy_fread == ny_phy);

  p = rs22_psf_2d_create(nx_phy_zp, ny_phy_zp);

  set0(p->h, p->N_phy);

  index = 0;
  for (j = 0; j < nx_phy; j++) {
    r = fread(&(p->h[index]), sizeof(elem), ny_phy, fid);
    assert(r == ny_phy);

    index += ny_phy_zp;
  }

  fclose(fid);

  return p;
}

void rs22_psf_2d_execute_r(const rs22_psf_2d *p_h, r_psf_2d *p_x) {
  int kx, ky;
  z_elem H_kx_ky;
  z_elem *X_kx_ky;

  assert(p_h);
  assert(p_x);
  assert(p_h->nx_log == p_x->nx_log);
  assert(p_h->ny_log == p_x->ny_log);
  assert(p_h->d == F);
  assert(p_x->d == F);


  for (ky = 0; ky < ((int) p_x->ny_phy/2); ky++) {
    for (kx = 0; kx < p_x->nx_phy; kx++) {
      rs22_psf_2d_dft_get(p_h, kx, ky, &H_kx_ky);
      X_kx_ky = (z_elem *) &(p_x->h[2*(ky + kx * ((int) p_x->ny_phy/2))]);
      zmul(X_kx_ky, (const z_elem *) &H_kx_ky);
    }
  }
}
