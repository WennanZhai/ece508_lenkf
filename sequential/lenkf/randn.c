#include <math.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <gsl/gsl_randist.h>

#include "randn.h"


boolean ROW_MAJOR = True;

boolean RANDN_DEBUG = False;

#define BUFFER_SIZE 256
static char RANDN_DIR[BUFFER_SIZE] = "/tmp";
static char RANDN_FILE[BUFFER_SIZE] = "randn";


/******************************************************************************/

static void randn_init_file(void);
static void randn_exit_file(void);
static void randn_reset_file(void);
static double randn_file(double sigma);

static void randn_init_gsl(void);
static void randn_exit_gsl(void);
static void randn_reset_gsl(void);
static double randn_gsl(double sigma);

static elem *randn_init_r_filter_new_r(r_filter_new *n, elem *h, int depth);

static void randn_v_conv_add_r_col_major(elem **u, int inc_u,
                                         r_filter_new *w, elem **w_h,
                                         int rank, int *n, int depth);

static void randn_v_conv_add_r_row_major(elem **u, int inc_u,
                                         r_filter_new *w, elem **w_h,
                                         int rank, int *n, int depth);

/******************************************************************************/



static FILE *randn_fid;

void randn_init_file(void) {
  char buffer[2*BUFFER_SIZE];

  strcat(strncpy(buffer, RANDN_DIR, BUFFER_SIZE), "/");
  strncat(buffer, RANDN_FILE, BUFFER_SIZE);

  randn_fid = fopen(buffer, "r");
  assert(randn_fid);
}

void randn_exit_file(void) {
  fclose(randn_fid);
}

void randn_reset_file(void) {
  rewind(randn_fid);
}

void randn_set_file(const char *file) {
  assert(file);

  strncpy(RANDN_FILE, file, BUFFER_SIZE);
}

void randn_set_dir(const char *dir) {
  assert(dir);

  strncpy(RANDN_DIR, dir, BUFFER_SIZE);
}

double randn_file(double sigma) {
  double n;
  int c;
  
  c = fread(&n, sizeof(double), 1, randn_fid);
  assert(c == 1);

  return sigma*n;
}


/******************************************************************************/

static gsl_rng *RANDN_RNG = NULL;
static unsigned long int SEED;

void randn_init_gsl(void) {
  RANDN_RNG = gsl_rng_alloc(gsl_rng_mt19937);
  assert(RANDN_RNG);

  SEED = time(0);
  randn_reset();
}

void randn_exit_gsl(void) {
  gsl_rng_free(RANDN_RNG);
}

void randn_reset_gsl(void) {
  assert(RANDN_RNG);
  
  gsl_rng_set(RANDN_RNG, SEED);
}

double randn_gsl(double sigma) {
  assert(RANDN_RNG);
  return gsl_ran_gaussian_ziggurat(RANDN_RNG, sigma);
}

/******************************************************************************/

void randn_init(void) {
  if (RANDN_DEBUG) {
    randn_init_file();
  }
  else {
    randn_init_gsl();
  }
}

void randn_exit(void) {
  if (RANDN_DEBUG) {
    randn_exit_file();
  }
  else {
    randn_exit_gsl();
  }
}

void randn_reset(void) {
  if (RANDN_DEBUG) {
    randn_reset_file();
  }
  else {
    randn_reset_gsl();
  }
}

double randn(double sigma) {
  if (RANDN_DEBUG) {
    return randn_file(sigma);
  }
  else {
    return randn_gsl(sigma);
  }
}

void randn_seed(unsigned long int seed) {
  if (RANDN_DEBUG) {
    fprintf(stderr, "seed not legal in debug mode\n");
    assert(0);
  }
  else {
    SEED = seed;
    randn_reset();
  }
}

/******************************************************************************/

/* Generate a random Gaussian vector with the prescribed covariance. */
void randn_v(vector *n, const sparse_rcs *C_sqrt) {
  int i;
  vector *v;
  
  assert(n);
  assert(C_sqrt);
  assert(C_sqrt->m == C_sqrt->n);
  assert(C_sqrt->m == n->n);

  v = vector_create(n->n);

  for (i = 0; i < v->n; i++) {
    v->v[i] = randn(1);
  }

  sparse_rcs_mvm(C_sqrt, v, n);
    
  vector_destroy(&v);
}

void randn_v_rect(vector *n, const sparse_rcs *A) {
  int i;
  vector *v;
  
  assert(n);
  assert(A);
  assert(n->n == A->m);

  v = vector_create(A->n);

  for (i = 0; i < v->n; i++) {
    v->v[i] = randn(1);
  }

  sparse_rcs_mvm(A, v, n);

  vector_destroy(&v);
}

/* NOTE: randn_v with the extra vector create and destroy + an eaxpy
   is faster than randn_v_add2col! */

/* Generate a random Gaussian vector with the prescribed covariance
   and add it to the jth column of A. n is ~ N(0,I). */
void randn_v_add2col(vector *n, full_r *A, int j, const sparse_rcs *C_sqrt) {
  int i;
  
  assert(n);
  assert(A);
  assert(C_sqrt);

  assert(C_sqrt->m == C_sqrt->n);
  assert(C_sqrt->m == n->n);
  assert(A->m == C_sqrt->n);
  assert(j >= 0 && j < A->n);

  for (i = 0; i < n->n; i++) {
    n->v[i] = randn(1);
  }

  sparse_rcs_mvm_add2col(C_sqrt, n, A, j);
}

/* Generate a random Gaussian vector with the prescribed covariance. */
/* The covariance matrix is assumed to be R_sqrt*R_sqrt' (and diagonal). */
void randn_v_diag(elem *v, const diag *R_sqrt) {
  int i;
  
  assert(v);
  assert(R_sqrt);
  assert(R_sqrt->m == R_sqrt->n);
  
  for (i = 0; i < R_sqrt->n; i++) {
    v[i] = randn(R_sqrt->v->v[i]);
  }
}

/* Generate a random Gaussian vector with the prescribed covariance. */
/* The covariance matrix is assumed to be */
/* R_sqrt(1:M_block,1:M_block)*R_sqrt(1:M_block,1:M_block)' (and diagonal). */
void randn_v_diag_block(elem *v, const elem *R_sqrt, const int inc_R_sqrt,
			const int M_block) {
  int i;
  
  assert(v);
  assert(R_sqrt);

  for (i = 0; i < M_block; i++) {
    v[i] = randn(R_sqrt[i*inc_R_sqrt]);
  }
}

/* Generate a matrix where each of its columns is a random vector with
 * the prescribed covariance.
 *
 * scratch must be the same size as n.
 */
void randn_m(full_r *n, const sparse_rcs *C_sqrt, full_r *scratch) {
  int N, P, i, k, n_elem, index, j_index;
  full_r *v;

  assert(n);
  assert(C_sqrt);
  assert(scratch);
  assert(C_sqrt->m == C_sqrt->n);
  assert(C_sqrt->m == n->m);
  assert(n->m == scratch->m);
  assert(n->n == scratch->n);
  
  
  N = n->m;
  P = n->n;

  v = scratch;

  for (i = 0; i < N*P; i++) {
    v->v_vector[i] = randn(1);
  }
  
  full_r_set0(n);

  for (i = 0; i < N; i++) {
    n_elem = C_sqrt->r[i+1] - C_sqrt->r[i];
    index = C_sqrt->r[i];

    for (k = 0; k < n_elem; k++) {
      j_index = C_sqrt->j[index];
      eaxpy(P, C_sqrt->v[index], &(v->v[j_index][0]), 1,
            &(n->v[i][0]), 1);
      index++;
    }
  }
}


static elem *randn_init_r_filter_new_r(r_filter_new *w, elem *h, int depth) {
  int i;
  
  assert(w);
  assert(h);
  assert(depth >= 0);
  assert(depth < w->rank);

  if (depth == w->rank - 1) {
    for (i = 0; i < w->n_log[w->rank-1]; i++) {
      h[i] = randn(1);
    }
    h += 2*((int) (floor(w->n_log[w->rank - 1]/2) + 1));
  }
  else {
    for (i = 0; i < w->n_log[depth]; i++) {
      h = randn_init_r_filter_new_r(w, h, depth+1);
    }
  }

  return h;
}


void randn_init_r_filter_new(r_filter_new *w) {
  assert(w);

  randn_init_r_filter_new_r(w, w->h, 0);
}


void randn_v_conv(r_filter_new *w, const r_filter_new *Q_sqrt) {
  assert(w);
  assert(Q_sqrt);
  assert(w->rank == Q_sqrt->rank);
  
  randn_init_r_filter_new(w);
  
  r_filter_new_dft(w);

  r_filter_new_execute_r(Q_sqrt, w);

  r_filter_new_idft(w);
}




void randn_v_conv_add(elem *u, int inc_u, r_filter_new *w,
		      const r_filter_new *Q_sqrt, int rank, int *n) {
  elem *u_ptr;
  elem *w_h;

  assert(u);
  assert(inc_u > 0);
  assert(w);
  assert(Q_sqrt);
  assert(n);

  randn_v_conv(w, Q_sqrt);

  u_ptr = u;
  w_h = w->h;

  if (ROW_MAJOR) {
    randn_v_conv_add_r_row_major(&u_ptr, inc_u, w, &w_h, rank, n, 0);
  }
  else {
    randn_v_conv_add_r_col_major(&u_ptr, inc_u, w, &w_h, rank, n, 0);
  }
}


static void randn_v_conv_add_r_col_major(elem **u, int inc_u,
                                         r_filter_new *w, elem **w_h,
                                         int rank, int *n, int depth) {
  int i;
  
  assert(u);
  assert(w);
  assert(depth >= 0);
  assert(depth < rank);

  if (rank == 1) {
    eaxpy(n[0], 1, *w_h, 1, *u, inc_u);
    return;
  }
  
  if (depth == rank - 1) {
    eaxpy(n[rank - 2], 1, *w_h, w->N_phy[rank - 1], *u, inc_u);
    *w_h += 1;
    *u += inc_u * n[rank - 2]; 
  }
  else if (depth == rank - 2) {
    for (i = 0; i < n[rank - 1]; i++) {
      randn_v_conv_add_r_col_major(u, inc_u, w, w_h, rank, n, depth+1);
    }

    *w_h -= n[rank - 1];
    *w_h += w->N_phy[rank - 2];
  }
  else {
    for (i = 0; i < n[depth]; i++) {
      randn_v_conv_add_r_col_major(u, inc_u, w, w_h, rank, n, depth+1);
    }
    
    *w_h += (w->n_log[depth] - n[depth]) * w->N_phy[depth+1];
  }
}


static void randn_v_conv_add_r_row_major(elem **u, int inc_u,
                                         r_filter_new *w, elem **w_h,
                                         int rank, int *n, int depth) {
  int i;
  
  assert(u);
  assert(w);
  assert(depth >= 0);
  assert(depth < rank);


  if (depth == rank - 1) {
    eaxpy(n[depth], 1, *w_h, 1, *u, inc_u);
    *w_h += w->N_phy[depth];
    *u += inc_u * n[depth];
  }
  else {
    for (i = 0; i < n[depth]; i++) {
      randn_v_conv_add_r_row_major(u, inc_u, w, w_h, rank, n, depth+1);
    }
    *w_h += (w->n_log[depth] - n[depth]) * w->N_phy[depth+1];
  }
}
