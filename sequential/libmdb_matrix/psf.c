#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "psf.h"
#include "eblas.h"


/********
 * c_psf
 ********/

c_psf *c_psf_create(int ny, int nx) {
  c_psf *p;

  assert(nx > 0);
  assert(ny > 0);

  p = malloc(sizeof(c_psf));
  assert(p);

  p->h = fftwe_malloc(sizeof(fftwe_complex) * nx * ny);
  assert(p->h);

  p->nx = nx;
  p->ny = ny;
  p->N = nx*ny;
  p->d = S;

  /* NOTE:

     ny and nx appear reversed here as compared to the discussion in
     the FFTW manual 3.1.  I mean for a psf to have ny rows and nx
     columns.  The first and second arguments of the plan functions
     define size of the first (# of rows in 2D - or the dimension
     whose index changes most slowly when cycling through p->h) and
     second (# of columns) dimensions of the psf, respectively.

   */

  p->forward_plan = fftwe_plan_dft_2d(ny, nx, p->h, p->h,
				      FFTW_FORWARD, FFTW_ESTIMATE);

  p->backward_plan = fftwe_plan_dft_2d(ny, nx, p->h, p->h,
				       FFTW_BACKWARD, FFTW_ESTIMATE);

  return p;
}

void c_psf_destroy(c_psf **p) {
  assert(p);
  assert(*p);

  fftwe_destroy_plan((*p)->forward_plan);
  fftwe_destroy_plan((*p)->backward_plan);

  free((*p)->h);

  free(*p);
  *p = NULL;
}

void c_psf_dft(c_psf *p) {
  assert(p);

  fftwe_execute(p->forward_plan);
  p->d = fftwe_domain_flip(p->d);
}

void c_psf_idft(c_psf *p) {
  assert(p);
  assert(p->d == F);

  fftwe_execute(p->backward_plan);
  escal(2*p->N, ((elem) 1)/p->N, (elem *) p->h, 1);
  p->d = fftwe_domain_flip(p->d);
}

void c_psf_printf(c_psf *p) {
  int i, j;
  fftwe_complex *c_index;

  assert(p);

  c_index = p->h;
  for (i = 0; i < p->ny; i++) {
    for (j = 0; j < p->nx; j++) {
      printf_z_elem_s((const z_elem *) c_index);
      c_index++;
    }
    printf("\n");
  }
}

void c_psf_set_real0(c_psf *p) {
  assert(p);

  escal(p->N, 0, &(p->h[0][0]), 2);
}

void c_psf_set_imag0(c_psf *p) {
  assert(p);

  escal(p->N, 0, &(p->h[0][1]), 2);
}

void c_psf_set0(c_psf *p) {
  assert(p);

  escal(2*p->N, 0, &(p->h[0][0]), 1);
}


/******************************************************************************/

/********
 * r_psf
 ********/

r_psf *r_psf_create(int ny, int nx) {
  r_psf *p;

  assert(nx > 0);
  assert(ny > 0);

  p = malloc(sizeof(r_psf));
  assert(p);

  p->nx_log = nx;
  p->ny_log = ny;
  p->N_log = nx*ny;

  /* See Section 2.4 of the FFTW manual 3.1 for a discussion of the
     necessary physical size of p->h to acoomodate an inplace
     transform. */
  p->ny_phy = ny;
  p->nx_phy = (floor(nx/2) + 1) * 2;
  p->N_phy = p->nx_phy * p->ny_phy;

  p->h = fftwe_malloc(sizeof(elem) * p->N_phy);

  p->d = S;

  p->forward_plan = fftwe_plan_dft_r2c_2d(ny, nx, p->h, (fftwe_complex *) p->h,
					  FFTW_ESTIMATE);

  p->backward_plan = fftwe_plan_dft_c2r_2d(ny, nx, (fftwe_complex *) p->h, p->h,
					   FFTW_ESTIMATE);

  return p;
}

void r_psf_destroy(r_psf **p) {
  assert(p);
  assert(*p);

  fftwe_destroy_plan((*p)->forward_plan);
  fftwe_destroy_plan((*p)->backward_plan);

  free((*p)->h);

  free(*p);
  *p = NULL;
}

void r_psf_dft(r_psf *p) {
  assert(p);

  fftwe_execute(p->forward_plan);
  p->d = fftwe_domain_flip(p->d);
}

void r_psf_idft(r_psf *p) {
  assert(p);
  assert(p->d == F);

  fftwe_execute(p->backward_plan);
  escal(p->N_phy, ((elem) 1)/p->N_log, p->h, 1);

  p->d = fftwe_domain_flip(p->d);
}

void r_psf_printf(r_psf *p) {
  int i, j;
  elem *e_ptr;
  z_elem *c_e_ptr;

  assert(p);

  if (p->d == S) {
    e_ptr = p->h;
    for (i = 0; i < p->ny_log; i++) {
      e_ptr = &(p->h[i*p->nx_phy]);
      for (j = 0; j < p->nx_log; j++) {
	printf_elem_s(*e_ptr);
	e_ptr++;
      }
      printf("\n");
    }
  }
  else if (p->d == F) {
    c_e_ptr = (z_elem *) p->h;
    for (i = 0; i < p->ny_log; i++) {
      for (j = 0; j < floor(p->nx_log/2) + 1; j++) {
	printf_z_elem_s((const z_elem *) c_e_ptr);
	c_e_ptr++;
      }
      printf("\n");
    }
  }
  else {
    assert(0);
  }
}

void r_psf_printf_dft(r_psf *p) {
  int i, j;
  z_elem H_ky_kx;

  assert(p);
  assert(p->d == F);

  for (i = 0; i < p->ny_log; i++) {
    for (j = 0; j < p->nx_log; j++) {
      r_psf_dft_get(p, i, j, &H_ky_kx);
      printf_z_elem_s((const z_elem *) &H_ky_kx);
    }
    printf("\n");
  }
}

void r_psf_set0(r_psf *p) {
  assert(p);

  set0(p->h, p->N_phy);
}

void r_psf_dft_get(const r_psf *p, int ky, int kx, z_elem *H_ky_kx) {
  z_elem *X_k;
  int nx_log_half;
  int nx_phy_half;
  int row, col, index;

  assert(p);
  assert(H_ky_kx);
  assert(ky >= 0 && ky < p->ny_log);
  assert(kx >= 0 && kx < p->nx_log);
  assert(p->d == F);


  nx_log_half = floor(p->nx_log/2);
  nx_phy_half = floor(p->nx_phy/2);

  if (kx <= nx_log_half) {
    col = kx;
    row = ky;
  }
  else {
    col = p->nx_log - kx;
    row = abs(p->ny_log - ky) % p->ny_log;
  }

  index = 2*(row*nx_phy_half + col);
  assert(index >= 0 && index < p->N_phy - 1);

  X_k = (z_elem *) &(p->h[index]);

  (*H_ky_kx)[0] = (*X_k)[0];

  if (kx <= nx_log_half) {
    (*H_ky_kx)[1] = (*X_k)[1];
  }
  else {
    (*H_ky_kx)[1] = -(*X_k)[1];
  }
}

void r_psf_execute_r(const r_psf *p_h, r_psf *p_x) {
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

/***********
 * rs22_psf
 ***********/

rs22_psf *rs22_psf_create(int ny_phy, int nx_phy) {
  rs22_psf *p;

  assert(ny_phy > 0);
  assert(nx_phy > 0);

  p = malloc(sizeof(rs22_psf));
  assert(p);

  p->ny_phy = ny_phy;
  p->nx_phy = nx_phy;
  p->N_phy = ny_phy * nx_phy;

  p->ny_log = 2*ny_phy;
  p->nx_log = 2*nx_phy;
  p->N_log = 4 * ny_phy * nx_phy;

  p->h = fftwe_malloc(sizeof(elem) * ny_phy * nx_phy);
  assert(p->h);

  p->d = S;

  p->forward_plan = fftwe_plan_r2r_2d(ny_phy, nx_phy, p->h, p->h,
				      FFTW_REDFT10, FFTW_REDFT10,
				      FFTW_ESTIMATE);

  p->backward_plan = fftwe_plan_r2r_2d(ny_phy, nx_phy, p->h, p->h,
				       FFTW_REDFT01, FFTW_REDFT01,
				       FFTW_ESTIMATE);

  return p;
}

void rs22_psf_destroy(rs22_psf **p) {
  assert(p);
  assert(*p);

  fftwe_free((*p)->h);

  fftwe_destroy_plan((*p)->forward_plan);
  fftwe_destroy_plan((*p)->backward_plan);

  free(*p);
  *p = NULL;
}

void rs22_psf_dft(rs22_psf *p) {
  assert(p);

  fftwe_execute(p->forward_plan);
  p->d = fftwe_domain_flip(p->d);
}

void rs22_psf_idft(rs22_psf *p) {
  assert(p);
  assert(p->d == F);

  fftwe_execute(p->backward_plan);
  escal(p->N_phy, ((elem) 1) / p->N_log, p->h, 1);

  p->d = fftwe_domain_flip(p->d);
}

void rs22_psf_printf(rs22_psf *p) {
  int i, j;
  elem *e_ptr;

  assert(p);

  e_ptr = p->h;
  for (i = 0; i < p->ny_phy; i++) {
    for (j = 0; j < p->nx_phy; j++) {
      printf_elem_s(*e_ptr);
      e_ptr++;
    }
    printf("\n");
  }
}

void rs22_psf_printf_dft(rs22_psf *p) {
  int i, j;
  z_elem H_ky_kx;

  assert(p);
  assert(p->d == F);

  for (i = 0; i < p->ny_log; i++) {
    for (j = 0; j < p->nx_log; j++) {
      rs22_psf_dft_get(p, i, j, &H_ky_kx);
      printf_z_elem_s((const z_elem *) &H_ky_kx);
    }
    printf("\n");
  }
}

void rs22_psf_dft_get(const rs22_psf *p, int ky, int kx,
			     z_elem *H_ky_kx) {
  z_elem z;
  elem theta;
  int row, index;

  assert(p);
  assert(H_ky_kx);
  assert(ky >= 0 && ky < p->ny_log);
  assert(kx >= 0 && kx < p->nx_log);


  if (ky == 0) {
    row = 0;

    (*H_ky_kx)[0] = 1;
    (*H_ky_kx)[1] = 0;
  }
  else if (ky < p->ny_phy) {
    row = ky;

    theta = M_PI * ky / (2 * p->ny_phy);
    expj(theta, H_ky_kx);
  }
  else if (ky == p->ny_phy) {
    (*H_ky_kx)[0] = 0;
    (*H_ky_kx)[1] = 0;

    return;
  }
  else {
    row = 2 * p->ny_phy - ky;

    theta = M_PI * ky / (2 * p->ny_phy);
    expj(theta, H_ky_kx);
    (*H_ky_kx)[0] *= -1;
    (*H_ky_kx)[1] *= -1;
  }

  if (kx == 0) {
    index = row * p->nx_phy;

    (*H_ky_kx)[0] *= p->h[index];
    (*H_ky_kx)[1] *= p->h[index];
  }
  else if (kx < p->nx_phy) {
    theta = M_PI * kx / (2 * p->nx_phy);
    expj(theta, &z);

    zmul(H_ky_kx, (const z_elem *) &z);

    index = row * p->nx_phy + kx;

    (*H_ky_kx)[0] *= p->h[index];
    (*H_ky_kx)[1] *= p->h[index];
  }
  else if (kx == p->nx_phy) {
    (*H_ky_kx)[0] = 0;
    (*H_ky_kx)[1] = 0;

    index = row * p->nx_phy + kx;

    return;
  }
  else {
    theta = M_PI * kx / (2 * p->nx_phy);
    expj(theta, &z);

    zmul(H_ky_kx, (const z_elem *) &z);

    index = row * p->nx_phy + 2 * p->nx_phy - kx;

    (*H_ky_kx)[0] *= -p->h[index];
    (*H_ky_kx)[1] *= -p->h[index];
  }
}

rs22_psf *rs22_psf_import(char *filename) {
  FILE *fid;
  int sizeof_elem;
  int ny_phy, nx_phy;
  rs22_psf *p;
  int r;

  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  r = fread(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);
  assert(sizeof_elem == sizeof(elem));

  r = fread(&ny_phy, sizeof(int), 1, fid);
  assert(r == 1);

  r = fread(&nx_phy, sizeof(int), 1, fid);
  assert(r == 1);

  p = rs22_psf_create(ny_phy, nx_phy);

  r = fread(p->h, sizeof(elem), p->N_phy, fid);
  assert(r == p->N_phy);

  fclose(fid);

  return p;
}

rs22_psf *rs22_psf_import_zp(const char *filename, int ny_phy, int nx_phy,
			     int ny_phy_zp, int nx_phy_zp) {
  FILE *fid;
  int sizeof_elem;
  int ny_phy_fread, nx_phy_fread;
  rs22_psf *p;
  int i, index;
  int r;

  assert(filename);
  assert(ny_phy > 0);
  assert(nx_phy > 0);
  assert(ny_phy_zp > 0 && ny_phy_zp >= ny_phy);
  assert(nx_phy_zp > 0 && nx_phy_zp >= nx_phy);

  fid = fopen(filename, "r");
  assert(fid);

  r = fread(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);
  assert(sizeof_elem == sizeof(elem));

  r = fread(&ny_phy_fread, sizeof(int), 1, fid);
  assert(r == 1);

  r = fread(&nx_phy_fread, sizeof(int), 1, fid);
  assert(r == 1);

  assert(ny_phy_fread == ny_phy);
  assert(nx_phy_fread == nx_phy);

  p = rs22_psf_create(ny_phy_zp, nx_phy_zp);

  set0(p->h, p->N_phy);

  index = 0;
  for (i = 0; i < ny_phy; i++) {
    r = fread(&(p->h[index]), sizeof(elem), nx_phy, fid);
    assert(r == nx_phy);

    index += nx_phy_zp;
  }

  fclose(fid);

  return p;
}


void rs22_psf_export(const char *filename, rs22_psf *p) {
  FILE *fid;
  int r;
  int sizeof_elem;

  assert(filename);
  assert(p);


  fid = fopen(filename, "w");
  assert(fid);

  sizeof_elem = sizeof(elem);
  r = fwrite(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&p->ny_phy, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&p->nx_phy, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(p->h, sizeof(elem), p->N_phy, fid);
  assert(r == p->N_phy);

  fclose(fid);
}


void rs22_psf_execute_r(const rs22_psf *p_h, r_psf *p_x) {
  int ky, kx;
  z_elem H_ky_kx;
  z_elem *X_ky_kx;
  elem *e_ptr;

  assert(p_h);
  assert(p_x);
  assert(p_h->ny_log == p_x->ny_log);
  assert(p_h->nx_log == p_x->nx_log);
  assert(p_h->d == F);
  assert(p_x->d == F);


  for (ky = 0; ky < p_x->ny_log; ky++) {
    e_ptr = &(p_x->h[ky*p_x->nx_phy]);
    for (kx = 0; kx < ((int) p_x->nx_phy/2); kx++) {
      rs22_psf_dft_get(p_h, ky, kx, &H_ky_kx);
      X_ky_kx = (z_elem *) e_ptr;
      zmul(X_ky_kx, (const z_elem *) &H_ky_kx);
      e_ptr += 2;
    }
  }
}
