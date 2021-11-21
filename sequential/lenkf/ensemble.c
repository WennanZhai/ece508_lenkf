#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <gsl/gsl_randist.h>

#include "ensemble.h"
#include "randn.h"



ensemble *ensemble_create(int N, int L) {
  ensemble *e;

  e = malloc(sizeof(ensemble));
  assert(e);

  e->L = L;
  e->N = N;

  e->X = full_r_create(N, L);

  return e;
}


void ensemble_destroy(ensemble **e) {
  assert(*e);

  full_r_destroy(&(*e)->X);
  free(*e);

  *e = NULL;
}


/* Compute the ensemble mean. */
/* mean <- mean of the ensemble e */
void ensemble_mean(const ensemble *e, vector *mean) {
  int i;
  const elem one = 1;
  
  assert(e);
  assert(mean);
  assert(mean->n == e->N);
  
  for (i = 0; i < e->N; i++) {
    mean->v[i] = edot(e->L, &(e->X->v[i][0]), 1, &one, 0);
  }
  
  escal(mean->n, ((elem) 1/(e->L)), mean->v, 1);
}


/* Compute the ensemble of perterbations about the mean. */
/* e <- e - mean */
void ensemble_subtract_mean(ensemble *e, vector *mean) {
  int i;
  
  assert(e);
  assert(mean);
  assert(e->N == mean->n);
  
  for (i = 0; i < e->N; i++) {
    eaxpy(e->L, -1, &(mean->v[i]), 0, &(e->X->v[i][0]), 1);
  }
}


/* Copy ensemble. */
/* dest <- source */
void ensemble_copy(ensemble *dest, const ensemble *source) {
  assert(dest);
  assert(source);

  assert(dest->L == source->L);
  assert(dest->N == source->N);

  ecopy(dest->L * dest->N, source->X->v_vector, 1, dest->X->v_vector, 1);
}


ensemble *ensemble_import(const char *filename) {
  ensemble *e;
  full_r *X;
  
  assert(filename);

  X = full_r_import(filename);

  e = malloc(sizeof(ensemble));
  assert(e);

  e->N = X->m;
  e->L = X->n;
  e->X = X;
  
  return e;
}


void ensemble_export(const char *filename, const ensemble *e) {
  assert(filename);
  assert(e);

  full_r_export(filename, e->X);
}


void ensemble_printf(const ensemble *e) {
  ensemble_fprintf(stdout, e);
}


void ensemble_fprintf(FILE *fid, const ensemble *e) {
  int i, l;

  assert(fid);
  assert(e);

  for (i = 0; i < e->N; i++) {
    for (l = 0; l < e->L; l++) {
      fprintf_elem_s(fid, e->X->v[i][l]);
    }

    fprintf(fid, "\n");
  }
}


/* Set all elements of the ensemble to 0 */
void ensemble_set0(ensemble *e) {
  assert(e);

  escal(e->L * e->N, 0, e->X->v_vector, 1);
}


/* Add noise with given covariance to all members of the ensemble. */
/* x^l <- x^l + u^l   u^l ~ N(0, Q) */
void ensemble_add_noise(ensemble *e, const sparse_rcs *Q_sqrt,
			vector *scratch) {
  int l;

  assert(e);
  assert(Q_sqrt);
  assert(scratch);
  assert(scratch->n == e->N);

  for (l = 0; l < e->L; l++) {
    randn_v(scratch, Q_sqrt);
    eaxpy(scratch->n, 1, scratch->v, 1, &(e->X->v[0][l]), e->L);
  }
}


void ensemble_add_noise_ab(arg_bundle *ab, const sparse_rcs *Q_sqrt) {
  assert(ab);
  ensemble_add_noise(ab->e, Q_sqrt, ab->scratch);
}


/* Add noise with given covariance to all members of the ensemble
   using the convolutional technique. */
/* x^l <- x^l + u^l   u^l ~ N(0, Q) */
void ensemble_add_noise_conv(ensemble *e, const r_filter_new *Q_sqrt,
			     r_filter_new *u_conv, int rank, int *n) {
  int l;
  
  assert(e);
  assert(Q_sqrt);
  assert(u_conv);
  
  for (l = 0; l < e->L; l++) {
    randn_v_conv_add(&e->X->v[0][l], e->L, u_conv, Q_sqrt, rank, n);
  }
}


void ensemble_add_noise_conv_ab(arg_bundle *ab, const r_filter_new *Q_sqrt) {
  assert(ab);
  
  ensemble_add_noise_conv(ab->e, Q_sqrt, ab->u_conv,
			  ab->config->rank, ab->config->n);
}


/* Compute the covariance of the ensemble */
void ensemble_cov(ensemble *e, full_r *C) {
  vector *mean;
  
  assert(e);
  assert(C);

  assert(C->m == e->N);
  assert(C->n == e->N);

  mean = vector_create(e->N);

  ensemble_mean(e, mean);
  ensemble_subtract_mean(e, mean);

  /* C <- X_p * X_p' */
  egemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        e->N, e->N, e->L, 1,
        e->X->v_vector, e->L,
        e->X->v_vector, e->L,
        0, C->v_vector, C->n);

  /* C <- 1/(L-1) * C */
  escal(C->m * C->n, ((elem) 1)/(e->L-1), C->v_vector, 1);
        
  vector_destroy(&mean);
}


/* Add mean vector to each ensemble member */
void ensemble_add_mean(ensemble *e, const vector *mean) {
  int i;
  
  assert(e);
  assert(mean);
  assert(e->N == mean->n);
  
  for (i = 0; i < e->L; i++) {
    eaxpy(e->N, 1, mean->v, 1, &(e->X->v[0][i]), e->L);
  }
}


/* Initialize each ensemble member to the vector x0 and add N(0,P0)
   noise. */
void ensemble_init(arg_bundle *ab) {
  ensemble *e;
  vector *x0;
  sparse_rcs *PI_sqrt;
  vector *scratch;
  
  assert(ab);

  e = ab->e;
  x0 = ab->x_mean;
  assert(e->N == x0->n);

  assert(ab->config->randn_conv == False);
  PI_sqrt = ab->PI_sqrt.rcs;

  scratch = ab->scratch;

  ensemble_set0(e);
  
  ensemble_add_noise_ab(ab, PI_sqrt);
}


/* Initialize each ensemble member to the vector x0 and add N(0,PI_0)
   noise. */
void ensemble_init_conv(arg_bundle *ab) {
  ensemble *e;
  vector *x0;
  r_filter_new *PI_sqrt;
  
  assert(ab);

  e = ab->e;
  x0 = ab->x_mean;
  assert(e->N == x0->n);

  assert(ab->config->randn_conv == True);
  PI_sqrt = ab->PI_sqrt.filter;

  ensemble_set0(e);
  
  ensemble_add_noise_conv_ab(ab, PI_sqrt);
}


/* Compute the sample trace from the ensemble. */
elem ensemble_trace(const ensemble *e) {
  assert(e);
  
  return edot(e->N * e->L, (const elem *) e->X->v_vector, 1,
	      (const elem *) e->X->v_vector, 1) / (e->L - 1);
}


/* Compute the dot product X(i,:) * X(j,:)' */
elem ensemble_edot(const ensemble *e, const int i, const int j) {
  assert(e);
  assert(i >= 0 && i < e->N);
  assert(j >= 0 && j < e->N);

  return edot(e->L, &e->X->v[i][0], 1, &e->X->v[j][0], 1);
}
