#ifndef PSF_2D_H
#define PSF_2D_H


#include "fftwe.h"


/**********
 * c_psf_2d
 **********/

/* A complex point spread function. */

typedef struct {
  fftwe_complex *h;
  int nx;
  int ny;
  int N;
  domain d;

  fftwe_plan forward_plan;
  fftwe_plan backward_plan;
} c_psf_2d;

c_psf_2d *c_psf_2d_create(int nx, int ny);
void c_psf_2d_destroy(c_psf_2d **p);
void c_psf_2d_dft(c_psf_2d *p);
void c_psf_2d_idft(c_psf_2d *p);
void c_psf_2d_printf(c_psf_2d *p);
void c_psf_2d_set_real0(c_psf_2d *p);
void c_psf_2d_set_imag0(c_psf_2d *p);
void c_psf_2d_set0(c_psf_2d *p);


/**********
 * r_psf_2d
 **********/

/* A real point spread function. */

typedef struct {
  elem *h;
  int nx_phy;
  int ny_phy;
  int N_phy;
  int nx_log;
  int ny_log;
  int N_log;

  domain d;

  fftwe_plan forward_plan;
  fftwe_plan backward_plan;
} r_psf_2d;

r_psf_2d *r_psf_2d_create(int nx, int ny);
void r_psf_2d_destroy(r_psf_2d **p);
void r_psf_2d_dft(r_psf_2d *p);
void r_psf_2d_idft(r_psf_2d *p);
void r_psf_2d_printf(r_psf_2d *p);
void r_psf_2d_printf_dft(r_psf_2d *p);
void r_psf_2d_set0(r_psf_2d *p);
void r_psf_2d_dft_get(const r_psf_2d *p, int kx, int ky, z_elem *H_kx_ky);
void r_psf_2d_execute_r(const r_psf_2d *p_h, r_psf_2d *p_x);



/*************
 * rs22_psf_2d
 *************/

/* A real point spread function that is Type-II symmetric in both the
   x and y directions. */

typedef struct {
  elem *h;
  int nx_phy;
  int ny_phy;
  int N_phy;
  int nx_log;
  int ny_log;
  int N_log;

  domain d;

  fftwe_plan forward_plan;
  fftwe_plan backward_plan;
} rs22_psf_2d;

rs22_psf_2d *rs22_psf_2d_create(int nx_phy, int ny_phy);
rs22_psf_2d *rs22_psf_2d_import(char *filename);
rs22_psf_2d *rs22_psf_2d_import_zp(const char *filename, int nx_phy, int ny_phy,
				   int nx_phy_zp, int ny_phy_zp);
void rs22_psf_2d_destroy(rs22_psf_2d **p);
void rs22_psf_2d_dft(rs22_psf_2d *p);
void rs22_psf_2d_idft(rs22_psf_2d *p);
void rs22_psf_2d_printf(rs22_psf_2d *p);
void rs22_psf_2d_printf_dft(rs22_psf_2d *p);
void rs22_psf_2d_dft_get(const rs22_psf_2d *p, int kx, int ky,
			 z_elem *H_kx_ky);
void rs22_psf_2d_execute_r(const rs22_psf_2d *p_h, r_psf_2d *p_x);


#endif
