#ifndef FFTWE_H
#define FFTWE_H

#include <fftw3.h>

#include "elem.h"


#ifdef DOUBLE_ELEM

typedef fftw_complex fftwe_complex;
typedef fftw_plan fftwe_plan;

#elif defined FLOAT_ELEM

typedef fftwf_complex fftwe_complex;
typedef fftwf_plan fftwe_plan;

#else
#error ?
#endif

typedef enum {S, F} domain;

void *fftwe_malloc(size_t n);
void fftwe_free(void *p);

/******************************************************************************/

fftwe_plan fftwe_plan_dft_1d(int n, fftwe_complex *in, fftwe_complex *out,
			     int sign, unsigned flags);

fftwe_plan fftwe_plan_dft_r2c_1d(int n, elem *in, fftwe_complex *out,
				 unsigned flags);

fftwe_plan fftwe_plan_dft_c2r_1d(int n, fftwe_complex *in, elem *out,
				 unsigned flags);

fftwe_plan fftwe_plan_r2r_1d(int n, elem *in, elem *out,
			     fftw_r2r_kind kind, unsigned flags);

/******************************************************************************/

fftwe_plan fftwe_plan_dft_2d(int d1, int d2,
			     fftwe_complex *in, fftwe_complex *out,
			     int sign, unsigned flags);

fftwe_plan fftwe_plan_dft_r2c_2d(int d1, int d2, elem *in, fftwe_complex *out,
				 unsigned flags);

fftwe_plan fftwe_plan_dft_c2r_2d(int d1, int d2, fftwe_complex *in, elem *out,
				 unsigned flags);

fftwe_plan fftwe_plan_r2r_2d(int d1, int d2, elem *in, elem *out,
			     fftw_r2r_kind kind_d1, fftw_r2r_kind kind_d2,
			     unsigned flags);

/******************************************************************************/

fftwe_plan fftwe_plan_dft_3d(int d1, int d2, int d3,
			     fftwe_complex *in, fftwe_complex *out,
			     int sign, unsigned flags);

fftwe_plan fftwe_plan_dft_r2c_3d(int d1, int d2, int d3,
				 elem *in, fftwe_complex *out,
				 unsigned flags);

fftwe_plan fftwe_plan_dft_c2r_3d(int d1, int d2, int d3,
				 fftwe_complex *in, elem *out,
				 unsigned flags);

fftwe_plan fftwe_plan_r2r_3d(int d1, int d2, int d3,
			     elem *in, elem *out,
			     fftw_r2r_kind kind_d1,
			     fftw_r2r_kind kind_d2,
			     fftw_r2r_kind kind_d3,
			     unsigned flags);

/******************************************************************************/

fftwe_plan fftwe_plan_dft(int rank, const int *n,
			  fftwe_complex *in, fftwe_complex *out,
			  int sign, unsigned flags);

fftwe_plan fftwe_plan_dft_r2c(int rank, const int *n,
			      elem *in, fftwe_complex *out,
			      unsigned flags);

fftwe_plan fftwe_plan_dft_c2r(int rank, const int *n,
			      fftwe_complex *in, elem *out,
			      unsigned flags);

fftwe_plan fftwe_plan_r2r(int rank, const int *n,
			  elem *in, elem *out,
			  const fftw_r2r_kind *kind, unsigned flags);

/******************************************************************************/

void fftwe_destroy_plan(fftwe_plan plan);
void fftwe_execute(const fftwe_plan plan);

void fftwe_export_wisdom_to_file(FILE *output_file);
void fftwe_import_wisdom_from_file(FILE *input_file);

domain fftwe_domain_flip(const domain d);

void fftwe_cleanup(void);

void fftwe_complex_fprintf(FILE *fid, const fftwe_complex *v);
void fftwe_complex_fprintf_s(FILE *fid, const fftwe_complex *v);
void fftwe_complex_fprintf_n(FILE *fid, const fftwe_complex *v);

void fftwe_complex_printf(const fftwe_complex *v);
void fftwe_complex_printf_s(const fftwe_complex *v);
void fftwe_complex_printf_n(const fftwe_complex *v);

#endif
