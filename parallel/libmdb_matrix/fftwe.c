#include <stdlib.h>
#include <assert.h>

#include "fftwe.h"


/******************************************************************************/

#ifdef DOUBLE_ELEM

void *fftwe_malloc(size_t n) {
  return fftw_malloc(n);
}

void fftwe_free(void *p) {
  fftw_free(p);
}

/******************************************************************************/

fftwe_plan fftwe_plan_dft_1d(int n, fftwe_complex *in,
			     fftwe_complex *out,
			     int sign, unsigned flags) {
  return fftw_plan_dft_1d(n, in, out, sign, flags);
}

fftwe_plan fftwe_plan_dft_r2c_1d(int n, elem *in, fftwe_complex *out,
				 unsigned flags) {
  return fftw_plan_dft_r2c_1d(n, in, out, flags);
}

fftwe_plan fftwe_plan_dft_c2r_1d(int n, fftwe_complex *in, elem *out,
				 unsigned flags) {
  return fftw_plan_dft_c2r_1d(n, in, out, flags);
}

fftwe_plan fftwe_plan_r2r_1d(int n, elem *in, elem *out,
			     fftw_r2r_kind kind, unsigned flags) {
  return fftw_plan_r2r_1d(n, in, out, kind, flags);
}

/******************************************************************************/

fftwe_plan fftwe_plan_dft_2d(int d1, int d2,
			     fftwe_complex *in, fftwe_complex *out,
			     int sign, unsigned flags) {
  return fftw_plan_dft_2d(d1, d2, in, out, sign, flags);
}

fftwe_plan fftwe_plan_dft_r2c_2d(int d1, int d2,
				 elem *in, fftwe_complex *out,
				 unsigned flags) {
  return fftw_plan_dft_r2c_2d(d1, d2, in, out, flags);
}

fftwe_plan fftwe_plan_dft_c2r_2d(int d1, int d2,
				 fftwe_complex *in, elem *out,
				 unsigned flags) {
  return fftw_plan_dft_c2r_2d(d1, d2, in, out, flags);
}

fftwe_plan fftwe_plan_r2r_2d(int d1, int d2, elem *in, elem *out,
			     fftw_r2r_kind kind_d1,
			     fftw_r2r_kind kind_d2,
			     unsigned flags) {
  return fftw_plan_r2r_2d(d1, d2, in, out, kind_d1, kind_d2, flags);
}

/******************************************************************************/

fftwe_plan fftwe_plan_dft_3d(int d1, int d2, int d3,
			     fftwe_complex *in, fftwe_complex *out,
			     int sign, unsigned flags) {
  return fftw_plan_dft_3d(d1, d2, d3, in, out, sign, flags);
}

fftwe_plan fftwe_plan_dft_r2c_3d(int d1, int d2, int d3,
				 elem *in, fftwe_complex *out,
				 unsigned flags) {
  return fftw_plan_dft_r2c_3d(d1, d2, d3, in, out, flags);
}

fftwe_plan fftwe_plan_dft_c2r_3d(int d1, int d2, int d3,
				 fftwe_complex *in, elem *out,
				 unsigned flags) {
  return fftw_plan_dft_c2r_3d(d1, d2, d3, in, out, flags);
}

fftwe_plan fftwe_plan_r2r_3d(int d1, int d2, int d3, elem *in, elem *out,
			     fftw_r2r_kind kind_d1,
			     fftw_r2r_kind kind_d2,
			     fftw_r2r_kind kind_d3,
			     unsigned flags) {
  return fftw_plan_r2r_3d(d1, d2, d3, in, out,
			  kind_d1, kind_d2, kind_d3, flags);
}

/******************************************************************************/

fftwe_plan fftwe_plan_dft(int rank, const int *n,
			  fftwe_complex *in, fftwe_complex *out,
			  int sign, unsigned flags) {
  return fftw_plan_dft(rank, n, in, out, sign, flags);
}

fftwe_plan fftwe_plan_dft_r2c(int rank, const int *n,
			      elem *in, fftwe_complex *out,
			      unsigned flags) {
  return fftw_plan_dft_r2c(rank, n, in, out, flags);
}

fftwe_plan fftwe_plan_dft_c2r(int rank, const int *n,
			      fftwe_complex *in, elem *out,
			      unsigned flags) {
  return fftw_plan_dft_c2r(rank, n, in, out, flags);
}

fftwe_plan fftwe_plan_r2r(int rank, const int *n,
			  elem *in, elem *out,
			  const fftw_r2r_kind *kind, unsigned flags) {
  return fftw_plan_r2r(rank, n, in, out, kind, flags);
}

/******************************************************************************/

void fftwe_destroy_plan(fftwe_plan plan) {
  fftw_destroy_plan(plan);
}

void fftwe_execute(const fftwe_plan plan) {
  fftw_execute(plan);
}

void fftwe_export_wisdom_to_file(FILE *output_file) {
  fftw_export_wisdom_to_file(output_file);
}

void fftwe_import_wisdom_from_file(FILE *input_file) {
  int r;

  r = fftw_import_wisdom_from_file(input_file);
  assert(r == 1);
}

void fftwe_cleanup(void) {
  fftw_cleanup();
}



/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

#elif defined FLOAT_ELEM

void *fftwe_malloc(size_t n) {
  return fftwf_malloc(n);
}

void fftwe_free(void *p) {
  fftwf_free(p);
}

/******************************************************************************/

fftwe_plan fftwe_plan_dft_1d(int n, fftwe_complex *in,
			     fftwe_complex *out,
			     int sign, unsigned flags) {
  return fftwf_plan_dft_1d(n, in, out, sign, flags);
}

fftwe_plan fftwe_plan_dft_r2c_1d(int n, elem *in, fftwe_complex *out,
					unsigned flags) {
  return fftwf_plan_dft_r2c_1d(n, in, out, flags);
}

fftwe_plan fftwe_plan_dft_c2r_1d(int n, fftwe_complex *in, elem *out,
				 unsigned flags) {
  return fftwf_plan_dft_c2r_1d(n, in, out, flags);
}

fftwe_plan fftwe_plan_r2r_1d(int n, elem *in, elem *out,
			     fftw_r2r_kind kind, unsigned flags) {
  return fftwf_plan_r2r_1d(n, in, out, kind, flags);
}

/******************************************************************************/

fftwe_plan fftwe_plan_dft_2d(int d1, int d2,
			     fftwe_complex *in, fftwe_complex *out,
			     int sign, unsigned flags) {
  return fftwf_plan_dft_2d(d1, d2, in, out, sign, flags);
}

fftwe_plan fftwe_plan_dft_r2c_2d(int d1, int d2,
				 elem *in, fftwe_complex *out,
				 unsigned flags) {
  return fftwf_plan_dft_r2c_2d(d1, d2, in, out, flags);
}

fftwe_plan fftwe_plan_dft_c2r_2d(int d1, int d2,
				 fftwe_complex *in, elem *out,
				 unsigned flags) {
  return fftwf_plan_dft_c2r_2d(d1, d2, in, out, flags);
}

fftwe_plan fftwe_plan_r2r_2d(int d1, int d2, elem *in, elem *out,
			     fftw_r2r_kind kind_d1,
			     fftw_r2r_kind kind_d2,
			     unsigned flags) {
  return fftwf_plan_r2r_2d(d1, d2, in, out, kind_d1, kind_d2, flags);
}

/******************************************************************************/

fftwe_plan fftwe_plan_dft_3d(int d1, int d2, int d3,
			     fftwe_complex *in, fftwe_complex *out,
			     int sign, unsigned flags) {
  return fftwf_plan_dft_3d(d1, d2, d3, in, out, sign, flags);
}

fftwe_plan fftwe_plan_dft_r2c_3d(int d1, int d2, int d3,
				 elem *in, fftwe_complex *out,
				 unsigned flags) {
  return fftwf_plan_dft_r2c_3d(d1, d2, d3, in, out, flags);
}

fftwe_plan fftwe_plan_dft_c2r_3d(int d1, int d2, int d3,
				 fftwe_complex *in, elem *out,
				 unsigned flags) {
  return fftwf_plan_dft_c2r_3d(d1, d2, d3, in, out, flags);
}

fftwe_plan fftwe_plan_r2r_3d(int d1, int d2, int d3, elem *in, elem *out,
			     fftw_r2r_kind kind_d1,
			     fftw_r2r_kind kind_d2,
			     fftw_r2r_kind kind_d3,
			     unsigned flags) {
  return fftwf_plan_r2r_3d(d1, d2, d3, in, out,
			   kind_d1, kind_d2, kind_d3, flags);
}

/******************************************************************************/

fftwe_plan fftwe_plan_dft(int rank, const int *n,
			  fftwe_complex *in, fftwe_complex *out,
			  int sign, unsigned flags) {
  return fftwf_plan_dft(rank, n, in, out, sign, flags);
}

fftwe_plan fftwe_plan_dft_r2c(int rank, const int *n,
			      elem *in, fftwe_complex *out,
			      unsigned flags) {
  return fftwf_plan_dft_r2c(rank, n, in, out, flags);
}

fftwe_plan fftwe_plan_dft_c2r(int rank, const int *n,
			      fftwe_complex *in, elem *out,
			      unsigned flags) {
  return fftwf_plan_dft_c2r(rank, n, in, out, flags);
}

fftwe_plan fftwe_plan_r2r(int rank, const int *n,
			  elem *in, elem *out,
			  const fftw_r2r_kind *kind, unsigned flags) {
  return fftwf_plan_r2r(rank, n, in, out, kind, flags);
}

/******************************************************************************/

void fftwe_destroy_plan(fftwe_plan plan) {
  fftwf_destroy_plan(plan);
}

void fftwe_execute(const fftwe_plan plan) {
  fftwf_execute(plan);
}

void fftwe_export_wisdom_to_file(FILE *output_file) {
  fftwf_export_wisdom_to_file(output_file);
}

void fftwe_import_wisdom_from_file(FILE *input_file) {
  int r;

  r = fftwf_import_wisdom_from_file(input_file);
  assert(r == 1);
}

void fftwe_cleanup(void) {
  fftwf_cleanup();
}
  
#else
#error ?
#endif


domain fftwe_domain_flip(const domain d) {
  if (d == S) {
    return F;
  }
  else if(d == F) {
    return S;
  }
  else {
    assert(0);
    exit(0);
  }
}


void fftwe_complex_fprintf(FILE *fid, const fftwe_complex *v) {
  fprintf_z_elem(fid, (const z_elem *) v);
}

void fftwe_complex_fprintf_s(FILE *fid, const fftwe_complex *v) {
  fprintf_z_elem_s(fid, (const z_elem *) v);
}

void fftwe_complex_fprintf_n(FILE *fid, const fftwe_complex *v) {
  fprintf_z_elem_n(fid, (const z_elem *) v);
}

void fftwe_complex_printf(const fftwe_complex *v) {
  fftwe_complex_fprintf(stdout, v);
}

void fftwe_complex_printf_s(const fftwe_complex *v) {
  fftwe_complex_fprintf_s(stdout, v);
}

void fftwe_complex_printf_n(const fftwe_complex *v) {
  fftwe_complex_fprintf_n(stdout, v);
}
