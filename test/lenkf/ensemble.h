#ifndef ENSEMBLE_H
#define ENSEMBLE_H

#include <stdio.h>

#ifdef LENKF_FLOAT_ELEM
#include "mdb_matrix_s.h"
#elif defined LENKF_DOUBLE_ELEM
#include "mdb_matrix_d.h"
#else
#error ?
#endif

#include "arg_bundle.h"

/* Note that X is stored row-wise.  However, it might make more sense
   to store X column-wise, i.e., each ensemble member x^l is stored
   contiguously on each column.  In the row-wise format, the first
   element of all ensemble members are stored contiguously, then the
   second element, and so on.  Row-wise storage was chosen because the
   most expensive step of the LEnKF is the computation of the
   localized sample covariance, which numerous dot products over the
   rows of the ensemble.  When L << N, column-wise storage would
   result in a large stride between members on rows, and the
   dot-product would not benefit from data locality.  However, this
   may not be so important when the localization is aggressive.  It
   might be interesting to generalize the code such that row-wise and
   column-wise storage implementations could be compared under
   different conditions, but this would require a significant
   restructuring of the code (code outside of ensemble.c assumes
   row-wise storage and operates on the ensemble directly). */


typedef struct ensemble_struct {
  int N;             /* dim of each member of the ensemble */
  int L;             /* # of members in the ensemble */

  full_r *X;
} ensemble;


ensemble *ensemble_create(int N, int L);
void ensemble_destroy(ensemble **e);
void ensemble_init(arg_bundle *ab);
void ensemble_init_conv(arg_bundle *ab);

ensemble *ensemble_import(const char *filename);
void ensemble_export(const char *filename, const ensemble *e);

void ensemble_copy(ensemble *y, const ensemble *x);
void ensemble_printf(const ensemble *x);
void ensemble_fprintf(FILE *fid, const ensemble *x);
void ensemble_set0(ensemble *e);

void ensemble_mean(const ensemble *e, vector *mean);
void ensemble_cov(ensemble *e, full_r *C);
void ensemble_subtract_mean(ensemble *e, vector *mean);
void ensemble_add_mean(ensemble *e, const vector *mean);
elem ensemble_trace(const ensemble *e);
elem ensemble_edot(const ensemble *e, const int i, const int j);

void ensemble_add_noise(ensemble *e, const sparse_rcs *Q_sqrt, vector *scratch);
void ensemble_add_noise_conv(ensemble *e, const r_filter_new *Q_sqrt,
			     r_filter_new *u_conv, int rank, int *n);

void ensemble_add_noise_ab(arg_bundle *ab, const sparse_rcs *Q_sqrt);
void ensemble_add_noise_conv_ab(arg_bundle *ab, const r_filter_new *Q_sqrt);

#endif
