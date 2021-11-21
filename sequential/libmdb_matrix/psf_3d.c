#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "psf_3d.h"
#include "fftwe.h"
#include "eblas.h"


/******************************************************************************/

/**********
 * c_psf_3d
 **********/

c_psf_3d *c_psf_3d_create(int nx, int ny, int nz) {
  c_psf_3d *p;

  assert(nx > 0);
  assert(ny > 0);
  assert(nz > 0);

  p = malloc(sizeof(c_psf_3d));
  assert(p);

  p->h = fftwe_malloc(sizeof(fftwe_complex) * nx * ny * nz);
  assert(p->h);

  p->nx = nx;
  p->ny = ny;
  p->nz = nz;

  p->N = nx * ny * nz;
  p->d = S;

  p->forward_plan = fftwe_plan_dft_3d(nx, ny, nz, p->h, p->h,
				      FFTW_FORWARD, FFTW_ESTIMATE);

  p->backward_plan = fftwe_plan_dft_3d(nx, ny, nz, p->h, p->h,
				       FFTW_BACKWARD, FFTW_ESTIMATE);

  return p;
}

void c_psf_3d_destroy(c_psf_3d **p) {
  assert(p);
  assert(*p);

  fftwe_destroy_plan((*p)->forward_plan);
  fftwe_destroy_plan((*p)->backward_plan);

  free((*p)->h);

  free(*p);
  *p = NULL;
}

void c_psf_3d_dft(c_psf_3d *p) {
  assert(p);

  fftwe_execute(p->forward_plan);
  p->d = fftwe_domain_flip(p->d);
}

void c_psf_3d_idft(c_psf_3d *p) {
  assert(p);
  assert(p->d == F);

  fftwe_execute(p->backward_plan);
  escal(3*p->N, ((elem) 1)/p->N, (elem *) p->h, 1);
  p->d = fftwe_domain_flip(p->d);
}

void c_psf_3d_printf(c_psf_3d *p) {
  int ix, iy, iz;

  assert(p);

  for (iz = 0; iz < p->nz; iz++) {
    for (iy = 0; iy < p->ny; iy++) {
      for (ix = 0; ix < p->nx; ix++) {
	printf_z_elem_s((const z_elem *) &p->h[iz + p->nz*(iy + p->ny*ix)]);
      }
      printf("\n");
    }
    printf("-----\n");
  }
}

void c_psf_3d_set_real0(c_psf_3d *p) {
  assert(p);

  escal(p->N, 0, &(p->h[0][0]), 2);
}

void c_psf_3d_set_imag0(c_psf_3d *p) {
  assert(p);

  escal(p->N, 0, &(p->h[0][1]), 2);
}

void c_psf_3d_set0(c_psf_3d *p) {
  assert(p);

  escal(2*p->N, 0, &(p->h[0][0]), 1);
}



/******************************************************************************/

/**********
 * r_psf_3d
 **********/

r_psf_3d *r_psf_3d_create(int nx, int ny, int nz) {
  r_psf_3d *p;

  assert(nx > 0);
  assert(ny > 0);
  assert(nz > 0);

  p = malloc(sizeof(r_psf_3d));
  assert(p);

  p->nx_log = nx;
  p->ny_log = ny;
  p->nz_log = nz;
  p->N_log = nx*ny*nz;

  /* See Section 2.4 of the FFTW manual 3.1 for a discussion of the
     necessary physical size of p->h to acoomodate an inplace
     transform. */
  p->nx_phy = nx;
  p->ny_phy = ny;
  p->nz_phy = (floor(nz/2) + 1) * 2;

  p->N_phy = p->nx_phy * p->ny_phy * p->nz_phy;

  p->h = fftwe_malloc(sizeof(elem) * p->N_phy);
  assert(p->h);

  p->d = S;

  p->forward_plan = fftwe_plan_dft_r2c_3d(nx, ny, nz,
					  p->h, (fftwe_complex *) p->h,
					  FFTW_ESTIMATE);

  p->backward_plan = fftwe_plan_dft_c2r_3d(nx, ny, nz,
					   (fftwe_complex *) p->h, p->h,
					   FFTW_ESTIMATE);

  return p;
}

void r_psf_3d_destroy(r_psf_3d **p) {
  assert(p);
  assert(*p);

  fftwe_destroy_plan((*p)->forward_plan);
  fftwe_destroy_plan((*p)->backward_plan);

  free((*p)->h);

  free(*p);
  *p = NULL;
}

void r_psf_3d_dft(r_psf_3d *p) {
  assert(p);

  fftwe_execute(p->forward_plan);
  p->d = fftwe_domain_flip(p->d);
}

void r_psf_3d_idft(r_psf_3d *p) {
  assert(p);
  assert(p->d == F);

  fftwe_execute(p->backward_plan);
  escal(p->N_phy, ((elem) 1)/p->N_log, p->h, 1);

  p->d = fftwe_domain_flip(p->d);
}

void r_psf_3d_printf(r_psf_3d *p) {
  int ix, iy, iz;

  assert(p);

  if (p->d == S) {
    for (iz = 0; iz < p->nz_log; iz++) {
      for (iy = 0; iy < p->ny_log; iy++) {
	for (ix = 0; ix < p->nx_log; ix++) {
	  printf_elem_s(p->h[iz + p->nz_phy*(iy + p->ny_phy*ix)]);
	}
	printf("\n");
      }
      printf("-----\n");
    }
  }
  else if (p->d == F) {
    for (iz = 0; iz < p->nz_phy/2; iz++) {
      for (iy = 0; iy < p->ny_phy; iy++) {
	for (ix = 0; ix < p->nx_log; ix++) {
	  printf_z_elem_s((const z_elem *)
			  &(p->h[2*(iz + p->nz_phy/2*(iy + p->ny_phy*ix))]));
	}
	printf("\n");
      }
      printf("-----\n");
    }
  }
  else {
    assert(0);
  }
}
