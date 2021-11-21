#ifndef PSF_H
#define PSF_H


#include "fftwe.h"


/********
 * c_psf
 ********/

/* A complex point spread function. */

typedef struct {
  fftwe_complex *h;
  int nx;
  int ny;
  int N;
  domain d;

  fftwe_plan forward_plan;
  fftwe_plan backward_plan;
} c_psf;

c_psf *c_psf_create(int ny, int nx);
void c_psf_destroy(c_psf **p);
void c_psf_dft(c_psf *p);
void c_psf_idft(c_psf *p);
void c_psf_printf(c_psf *p);
void c_psf_set_real0(c_psf *p);
void c_psf_set_imag0(c_psf *p);
void c_psf_set0(c_psf *p);


/********
 * r_psf
 ********/

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
} r_psf;

r_psf *r_psf_create(int ny, int nx);
void r_psf_destroy(r_psf **p);
void r_psf_dft(r_psf *p);
void r_psf_idft(r_psf *p);
void r_psf_printf(r_psf *p);
void r_psf_printf_dft(r_psf *p);
void r_psf_set0(r_psf *p);
void r_psf_dft_get(const r_psf *p, int ky, int kx, z_elem *H_ky_kx);
void r_psf_execute_r(const r_psf *p_h, r_psf *p_x);



/***********
 * rs22_psf
 ***********/

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
} rs22_psf;

rs22_psf *rs22_psf_create(int ny_phy, int nx_phy);
rs22_psf *rs22_psf_import(char *filename);
rs22_psf *rs22_psf_import_zp(const char *filename, int ny_phy, int nx_phy,
			     int ny_phy_zp, int nx_phy_zp);
void rs22_psf_export(const char *filename, rs22_psf *p);
void rs22_psf_destroy(rs22_psf **p);
void rs22_psf_dft(rs22_psf *p);
void rs22_psf_idft(rs22_psf *p);
void rs22_psf_printf(rs22_psf *p);
void rs22_psf_printf_dft(rs22_psf *p);
void rs22_psf_dft_get(const rs22_psf *p, int ky, int kx,
		      z_elem *H_ky_kx);
void rs22_psf_execute_r(const rs22_psf *p_h, r_psf *p_x);


#endif
