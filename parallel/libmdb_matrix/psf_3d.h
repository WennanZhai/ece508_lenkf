#ifndef PSF_3D_H
#define PSF_3D_H


#include "fftwe.h"


/***********
 * c_psf_3d
 ***********/

/* A complex 3d point spread function. */

typedef struct {
  fftwe_complex *h;
  int nx;
  int ny;
  int nz;
  int N;
  domain d;

  fftwe_plan forward_plan;
  fftwe_plan backward_plan;
} c_psf_3d;

c_psf_3d *c_psf_3d_create(int nx, int ny, int nz);
void c_psf_3d_destroy(c_psf_3d **p);
void c_psf_3d_dft(c_psf_3d *p);
void c_psf_3d_idft(c_psf_3d *p);
void c_psf_3d_printf(c_psf_3d *p);
void c_psf_3d_set_real0(c_psf_3d *p);
void c_psf_3d_set_imag0(c_psf_3d *p);
void c_psf_3d_set0(c_psf_3d *p);



/**********
 * r_psf_3d
 **********/

/* A real 3d point spread function. */

typedef struct {
  elem *h;
  int nx_phy;
  int ny_phy;
  int nz_phy;
  int N_phy;
  int nx_log;
  int ny_log;
  int nz_log;
  int N_log;

  domain d;

  fftwe_plan forward_plan;
  fftwe_plan backward_plan;
} r_psf_3d;

r_psf_3d *r_psf_3d_create(int nx, int ny, int nz);
void r_psf_3d_destroy(r_psf_3d **p);
void r_psf_3d_dft(r_psf_3d *p);
void r_psf_3d_idft(r_psf_3d *p);
void r_psf_3d_printf(r_psf_3d *p);
void r_psf_3d_printf_dft(r_psf_3d *p);
void r_psf_3d_set0(r_psf_3d *p);
void r_psf_3d_dft_get(const r_psf_3d *p, int kx, int ky, int kz,
		      z_elem *H_kx_ky_kz);
void r_psf_3d_execute_r(const r_psf_3d *p_h, r_psf_3d *p_x);


#endif
