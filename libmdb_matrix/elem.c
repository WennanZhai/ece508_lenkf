#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_sf_bessel.h>

#include "elem.h"
#include "eblas.h"


static void z_elem_2_gsl_complex(gsl_complex *g, const z_elem *z);
static void gsl_complex_2_z_elem(z_elem *z, const gsl_complex *g);

const elem ENAN = ((elem) 0) / ((elem) 0);


boolean e_isnan(elem v) {
  return v != v;
}


void fprintf_elem(FILE *fid, elem v) {
  assert(fid);

  if (fabse(v) < ELEM_EPSILON) {
    fprintf(fid, "%+f", 0.0);
  }
  else {
#ifdef DOUBLE_ELEM
    fprintf(fid, "%+f", v);
#elif defined FLOAT_ELEM
    fprintf(fid, "%+f", v);
#else
#error ?
#endif
  }
}


void fprintf_elem_s(FILE *fid, elem v) {
  assert(fid);

  fprintf_elem(fid, v);
  fputc((int) ' ', fid);
}


void fprintf_elem_n(FILE *fid, elem v) {
  assert(fid);

  fprintf_elem(fid, v);
  fputc((int) '\n', fid);
}


void printf_elem(elem v) {
  fprintf_elem(stdout, v);
}


void printf_elem_s(elem v) {
  fprintf_elem_s(stdout, v);
}


void printf_elem_n(elem v) {
  fprintf_elem_n(stdout, v);
}


void fprintf_z_elem(FILE *fid, const z_elem *v) {
  assert(fid);
  assert(v);

  fputc((int) '(', fid);
  fprintf_elem_s(fid, (*v)[0]);
  fprintf_elem(fid, (*v)[1]);
  fputc((int) ')', fid);
}


void fprintf_z_elem_s(FILE *fid, const z_elem *v) {
  fprintf_z_elem(fid, v);
  fputc((int) ' ', fid);
}


void fprintf_z_elem_n(FILE *fid, const z_elem *v) {
  fprintf_z_elem(fid, v);
  fputc((int) '\n', fid);
}


void printf_z_elem(const z_elem *v) {
  fprintf_z_elem(stdout, v);
}


void printf_z_elem_s(const z_elem *v) {
  fprintf_z_elem_s(stdout, v);
}


void printf_z_elem_n(const z_elem *v) {
  fprintf_z_elem_n(stdout, v);
}


elem sqrte(elem x) {
#ifdef DOUBLE_ELEM
  return sqrt(x);
#elif defined FLOAT_ELEM
  return (float) sqrt(x);
#else
#error ?
#endif
}


elem fabse(elem x) {
#ifdef DOUBLE_ELEM
  return fabs(x);
#elif defined FLOAT_ELEM
  return (float) fabs(x);
#else
#error ?
#endif
}


void set0(elem *v, int n) {
  assert(v);
  assert(n >= 0);

  escal(n, 0, v, 1);
}


void zset0(z_elem *v, int n) {
  z_elem zero;

  assert(v);
  assert(n >= 0);

  zero[0] = 0;
  zero[1] = 0;

  zescal(n, zero, v, 1);
}


elem sine(elem x) {
#ifdef DOUBLE_ELEM
  return sin(x);
#elif defined FLOAT_ELEM
  return (float) sin(x);
#else
#error ?
#endif
}


elem cose(elem x) {
#ifdef DOUBLE_ELEM
  return cos(x);
#elif defined FLOAT_ELEM
  return (float) cos(x);
#else
#error ?
#endif
}


/* y = exp(j*theta) */
void expj(elem theta, z_elem *y) {
  assert(y);

  (*y)[0] = cose(theta);
  (*y)[1] = sine(theta);
}


/* 2D jinc function */
elem jinc(elem x, elem y) {
  return jinc1d(sqrte(y*y + x*x));
}


/* 1D jinc function */
elem jinc1d(elem x) {
  if (fabse(x) < ELEM_EPSILON) {
    return M_PI/2;
  }
  else {
    return gsl_sf_bessel_J1(M_PI*x) / (M_PI*x);
  }
}


/* y = y * x,  where y and x are complex */
void zmul(z_elem *y, const z_elem *x) {
  z_elem z;

  assert(y);
  assert(x);

  z[0] = (*y)[0];
  z[1] = (*y)[1];

  (*y)[0] = (*x)[0]*z[0] - (*x)[1]*z[1];
  (*y)[1] = (*x)[1]*z[0] + (*x)[0]*z[1];
}


/* y = sqrt(x) (the principal square root), where y and x are complex */
void zsqrt(z_elem *y, const z_elem *x) {
  elem r;

  assert(y);
  assert(x);

  r = sqrte((*x)[0]*(*x)[0] + (*x)[1]*(*x)[1]);

  (*y)[0] = sqrte((r + (*x)[0])/2);
  (*y)[1] = (*x)[1]/sqrte(2*(r + (*x)[0]));
}


void z_elem_2_gsl_complex(gsl_complex *g, const z_elem *z) {
  assert(g);
  assert(z);

  GSL_SET_COMPLEX(g, (*z)[0], (*z)[1]);
}


static void gsl_complex_2_z_elem(z_elem *z, const gsl_complex *g) {
  assert(z);
  assert(g);

  (*z)[0] = GSL_REAL(*g);
  (*z)[1] = GSL_IMAG(*g);
}


/* y = sin(x), where y and x are complex */
void zsin(z_elem *y, const z_elem *x) {
  gsl_complex a, b;

  assert(y);
  assert(x);

  /* y = sin(r) cosh(i) + j cos(r) sinh(i) */
  z_elem_2_gsl_complex(&a, x);
  b = gsl_complex_sin(a);
  gsl_complex_2_z_elem(y, &b);
}


/* y = cos(x), where y and x are complex */
void zcos(z_elem *y, const z_elem *x) {
  gsl_complex a, b;

  assert(y);
  assert(x);

  /* y = cos(r) cosh(i) - j sin(r) sinh(i) */
  z_elem_2_gsl_complex(&a, x);
  b = gsl_complex_cos(a);
  gsl_complex_2_z_elem(y, &b);
}


/* y = arcsin(x), where y and x are complex */
void zarcsin(z_elem *y, const z_elem *x) {
  gsl_complex a, b;

  assert(y);
  assert(x);

  /* y = 1/j log(j*x + sqrt(1 - x^2)) */
  z_elem_2_gsl_complex(&a, x);
  b = gsl_complex_arcsin(a);
  gsl_complex_2_z_elem(y, &b);
}


/* y = arccos(x), where y and x are complex */
void zarccos(z_elem *y, const z_elem *x) {
  gsl_complex a, b;

  assert(y);
  assert(x);

  /* y = 1/j log(x + j*sqrt(1 - x^2)) */
  z_elem_2_gsl_complex(&a, x);
  b = gsl_complex_arccos(a);
  gsl_complex_2_z_elem(y, &b);
}


int sgne(elem x) {
  if (x == 0) {
    return 0;
  }
  else if (x > 0) {
    return 1;
  }
  else {
    return -1;
  }
}
