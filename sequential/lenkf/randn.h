#ifndef RANDN_H
#define RANDN_H

#include <gsl/gsl_rng.h>

#ifdef LENKF_FLOAT_ELEM
#include "mdb_matrix_s.h"
#elif defined LENKF_DOUBLE_ELEM
#include "mdb_matrix_d.h"
#else
#error ?
#endif


void randn_init(void);
void randn_exit(void);
void randn_reset(void);
void randn_seed(unsigned long int seed);

void randn_set_file(const char *file);
void randn_set_dir(const char *dir);

double randn(double sigma);
void randn_m(full_r *n, const sparse_rcs *C_sqrt, full_r *scratch);
void randn_v(vector *n, const sparse_rcs *C_sqrt);
void randn_v_rect(vector *n, const sparse_rcs *A);
void randn_v_add2col(vector *n, full_r *A, int j, const sparse_rcs *C_sqrt);
void randn_v_diag(elem *v, const diag *R_sqrt);
void randn_v_diag_block(elem *v, const elem *R_sqrt, const int inc_R_sqrt,
			const int M_block);

void randn_init_r_filter_new(r_filter_new *w);
void randn_v_conv(r_filter_new *w, const r_filter_new *Q_sqrt);
void randn_v_conv_add(elem *u, int inc_u, r_filter_new *w,
		      const r_filter_new *Q_sqrt, int rank, int *n);

#endif

