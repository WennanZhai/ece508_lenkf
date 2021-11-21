#ifndef ELEM_H
#define ELEM_H

#include <stdio.h>
#include <float.h>

#include "boolean.h"


#ifndef M_PI
#define M_PI           3.14159265358979323846  /* pi */
#endif


#ifdef DOUBLE_ELEM
typedef double elem;
typedef double z_elem[2];
#define ELEM_EPSILON DBL_EPSILON
#elif defined FLOAT_ELEM
typedef float elem;
typedef float z_elem[2];
#define ELEM_EPSILON FLT_EPSILON
#else
#error No elem type defined!
#endif

extern const elem ENAN;


/* THESE SHOULD ALL BE FUNCITONS OF AN ELEM * !!! */

boolean e_isnan(elem v);

void fprintf_elem(FILE *fid, elem v);
void fprintf_elem_s(FILE *fid, elem v);
void fprintf_elem_n(FILE *fid, elem v);

void printf_elem(elem v);
void printf_elem_s(elem v);
void printf_elem_n(elem v);


void fprintf_z_elem(FILE *fid, const z_elem *v);
void fprintf_z_elem_s(FILE *fid, const z_elem *v);
void fprintf_z_elem_n(FILE *fid, const z_elem *v);

void printf_z_elem(const z_elem *v);
void printf_z_elem_s(const z_elem *v);
void printf_z_elem_n(const z_elem *v);


elem sqrte(elem x);
elem fabse(elem x);

int sgne(elem x);

elem sine(elem x);
elem cose(elem x);
void expj(elem theta, z_elem *y);

elem jinc(elem x, elem y);
elem jinc1d(elem x);

void zmul(z_elem *y, const z_elem *x);
void zsqrt(z_elem *y, const z_elem *x);

void zsin(z_elem *y, const z_elem *x);
void zcos(z_elem *y, const z_elem *x);

void zarcsin(z_elem *y, const z_elem *x);
void zarccos(z_elem *y, const z_elem *x);

void set0(elem *v, int n);
void zset0(z_elem *v, int n);

#endif
