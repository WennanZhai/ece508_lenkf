#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <limits.h>

#include "lenkf.h"
#include "ensemble.h"
#include "arg_bundle.h"
#include "randn.h"
#include "edot_table.h"


/******************************************************************************/

static cfg_bool_t LENKF_DEBUG;
static FILE *LENKF_DEBUG_FID;

#define BUFFER_SIZE 256

static multi_sw *SW;
static multi_sw *SW_MU_TU;

static int N_CLOCKS = 0;
static cfg_bool_t ENABLE_PROFILING = cfg_false;


/******************************************************************************/

/*
static void compute_P_HT(arg_bundle *ab, const sparse_rcs *H,
			 const int row_H, const char name_H,
			 const int n_rows);
*/

void compute_P_HT(arg_bundle *ab, const sparse_rcs *H,
                  const int row_H, const char name_H,
                  const int n_rows);

static void compute_H_P_HT(arg_bundle *ab, const sparse_rcs *H,
			   const int row_H, const char name_H,
			   const int n_rows);

static void compute_H_P_HT_plus_R(arg_bundle *ab,
				  const elem *R_sqrt, const int inc_R_sqrt,
				  const char name_H,
				  const int n_rows);

static void compute_V_minus_H_X(arg_bundle *ab, const sparse_rcs *H,
				const int row_H, const char name_H,
				const int n_rows);

static void compute_V(arg_bundle *ab, const elem *R_sqrt, const int inc_R_sqrt,
		      const int n_rows);

static void compute_chol_B(arg_bundle *ab, const int n_rows);

static void compute_B_rdiv_E(arg_bundle *ab, const int n_rows);

static void compute_e(arg_bundle *ab, const sparse_rcs *H, int row_H,
		      const int n_rows, const elem *y, const int inc_y);

static void compute_B_rdiv_e(arg_bundle *ab, const int n_rows);

static void compute_X_update(arg_bundle *ab);

static void compute_x_mean_update(arg_bundle *ab);

static void compute_measurement_update(arg_bundle *ab,
				       const int i,
				       const sparse_rcs *H, int row_H,
				       const char name_H,
				       const elem *y, const int inc_y,
				       const elem *R_sqrt, const int inc_R_sqrt,
				       const int n_rows);

static void compute_trace(arg_bundle *ab, int i);


static void compute_time_update(arg_bundle *ab);

/******************************************************************************/

static void ensemble_add_state_noise_wrapper(arg_bundle *ab);

static void ensemble_init_wrapper(arg_bundle *ab);

/******************************************************************************/

static double compute_time_per_iter(const struct timeval *start,
                                    const struct timeval *current,
                                    const int i);

static double time_elapsed(const struct timeval *start,
                           const struct timeval *current);

static void printf_time_info(const struct timeval *start_time,
			     unsigned int *buffer_strlen,
			     const int i, const int N_steps,
			     FILE *stamp_fid);

static void TIC(const int clock_index);
static void TOC(const int clock_index);

static void TIC_MU_TU(const int clock_index);
static void TOC_MU_TU(const int clock_index);

/******************************************************************************/

/* See Algorithm 1.2.3 in Golub and van Loan 3rd and 4th editions:
 * Symmetric storage Gaxpy */
/*
static void compute_P_HT(arg_bundle *ab, const sparse_rcs *H,
			 const int row_H, const char name_H,
			 const int n_rows) {
*/
void compute_P_HT(arg_bundle *ab, const sparse_rcs *H,
                  const int row_H, const char name_H,
                  const int n_rows) {
  int i, j_index, j;
  elem edot_ij, C_v;
  edot_table *table;

  const ensemble *e = ab->e;
  sb_toe_r_it *C_it = ab->C_it;
  const int N = ab->config->N;

  vector *H_P_col;
  sparse_lil *P_HT;
  sparse_rcs *P_HT_rcs;


  TIC(1);

  assert(H);
  assert(row_H >= 0 && row_H + n_rows - 1 < H->m);

  TIC(2);
  table = edot_table_create();
  TOC(2);

  if (ab->P_HT) {
    sparse_lil_destroy(&ab->P_HT);
  }

  P_HT = ab->P_HT = sparse_lil_create(N, n_rows);
  H_P_col = vector_create(ab->config->N);

  for (i = row_H; i < row_H + n_rows; i++) {
    vector_set0(H_P_col);

    for (j_index = H->r[i]; j_index < H->r[i+1]; j_index++) {
      j = H->j[j_index];

      sb_toe_r_nz_it_init(C_it, j);

      while (sb_toe_r_nz_it_has_next(C_it)) {
        
	      C_v = sb_toe_r_nz_it_next(C_it);

	      if (edot_table_find(table, j, C_it->j, e)) {
	        edot_ij = table->record->edot;
	      } else {
	        edot_ij = edot_table_add(table, j, C_it->j, e);
	      }

	      /* Matrix element indexed in [j][i] order because the H_P
	      matrix is stored column-wise */
	      H_P_col->v[C_it->j] += C_v * H->v[j_index] * edot_ij;
      }
    }
    for (j = 0; j < N; j++) {
      if (H_P_col->v[j] != 0) {
	      sparse_lil_append(P_HT, j, i - row_H, H_P_col->v[j]);
      }
    }
  }

  vector_destroy(&H_P_col);

  TIC(2);
  edot_table_destroy(&table);
  TOC(2);

  sparse_lil_scal(P_HT, ((elem) 1)/(e->L - 1));

  if (LENKF_DEBUG) {
    fprintf(LENKF_DEBUG_FID, "\nP_%cT:\n", name_H);
    P_HT_rcs = sparse_lil_2_rcs(P_HT);
    sparse_rcs_fprintf(LENKF_DEBUG_FID, P_HT_rcs);
    sparse_rcs_destroy(&P_HT_rcs);
  }

  TOC(1);
}


static void compute_H_P_HT(arg_bundle *ab, const sparse_rcs *H,
			   const int row_H, const char name_H,
			   const int n_rows) {
  int i, j_index, j;

  full_c *B = ab->B;

  sparse_lil *P_HT = ab->P_HT;
  sparse_lil_row_it P_HT_it;
  elem H_v, P_HT_v;

  TIC(3);

  assert(H);
  assert(row_H >= 0 && row_H + n_rows - 1 < H->m);
  assert(B->m == B->n);
  assert(B->m == ab->config->M_block);
  assert(P_HT);

  full_c_set0(B);

  for (i = row_H; i < row_H + n_rows; i++) {
    for (j_index = H->r[i]; j_index < H->r[i+1]; j_index++) {
      H_v = H->v[j_index];
      j = H->j[j_index];

      sparse_lil_row_it_init(P_HT, j, &P_HT_it);

      while (sparse_lil_row_it_has_next(&P_HT_it)) {
	P_HT_v = sparse_lil_row_it_next(&P_HT_it);

	if (P_HT_it.j < i - row_H) {
	  continue;
	}

	/* Matrix indexed in [j][i] because it is stored column-major */
  	B->v[P_HT_it.j][i-row_H] += H_v * P_HT_v;
      }
    }
  }

  if (LENKF_DEBUG) {
    fprintf(LENKF_DEBUG_FID, "\n%c_P_%cT:\n", name_H, name_H);
    full_c_submatrix_fprintf(LENKF_DEBUG_FID, B, n_rows, n_rows);
  }

  TOC(3);
}


static void compute_H_P_HT_plus_R(arg_bundle *ab,
				  const elem *R_sqrt, const int inc_R_sqrt,
				  const char name_H, const int n_rows) {
  int i, index_B;

  full_c *B = ab->B;

  TIC(4);

  index_B = 0;
  for (i = 0; i < n_rows; i++) {
    B->v[i][i] += R_sqrt[i*inc_R_sqrt] * R_sqrt[i*inc_R_sqrt];
  }

  if (LENKF_DEBUG) {
    fprintf(LENKF_DEBUG_FID, "\n%c_P_%cT + R:\n", name_H, name_H);
    full_c_submatrix_fprintf(LENKF_DEBUG_FID, B, n_rows, n_rows);
  }

  TOC(4);
}


static void compute_V(arg_bundle *ab, const elem *R_sqrt, const int inc_R_sqrt,
		      const int n_rows) {
  int l;

  const lenkf_config *config = ab->config;
  full_c *E = ab->E;

  TIC(5);

  for (l = 0; l < config->L; l++) {
    /* Matrix element indexed in [j][i] order because the E matrix is
       stored column-wise */
    randn_v_diag_block(&(E->v[l][0]), R_sqrt, inc_R_sqrt, n_rows);
  }

  if (LENKF_DEBUG) {
    fprintf(LENKF_DEBUG_FID, "\nV:\n");
    full_c_rows_fprintf(LENKF_DEBUG_FID, E, 0, n_rows-1);
  }

  TOC(5);
}


static void compute_V_minus_H_X(arg_bundle *ab, const sparse_rcs *H,
				const int row_H, const char name_H,
				const int n_rows) {
  int i, index_H;

  const lenkf_config *config = ab->config;
  const full_r *X = ab->e->X;
  full_c *E = ab->E;

  TIC(6);

  assert(H);
  assert(row_H >= 0 && row_H + n_rows - 1 < H->m);
  assert(E->m >= n_rows);
  assert(E->n == X->n);
  assert(H->n == X->m);


  for (i = row_H; i < row_H + n_rows; i++) {
    for (index_H = H->r[i]; index_H < H->r[i+1]; index_H++) {
      eaxpy(config->L, -H->v[index_H],
	    &(X->v[H->j[index_H]][0]), 1,
	    &(E->v[0][i-row_H]), E->m);
    }
  }

  if (LENKF_DEBUG) {
    fprintf(LENKF_DEBUG_FID, "\nV - %c_X:\n", name_H);
    full_c_rows_fprintf(LENKF_DEBUG_FID, E, 0, n_rows-1);
  }

  TOC(6);
}


static void compute_chol_B(arg_bundle *ab, const int n_rows) {
  int r;

  full_c *B = ab->B;

  TIC(7);

  assert(n_rows <= B->n);

  r = epotrf(CblasColMajor, 'U', n_rows, B->v_vector, B->n);
  assert(r == 0);

  if (LENKF_DEBUG) {
    fprintf(LENKF_DEBUG_FID, "\nchol(B):\n");
    full_c_submatrix_fprintf(LENKF_DEBUG_FID, B, n_rows, n_rows);
  }

  TOC(7);
}


static void compute_B_rdiv_E(arg_bundle *ab, const int n_rows) {
  int r;

  const full_c *B = ab->B;
  full_c *E = ab->E;

  TIC(8);

  assert(n_rows <= B->n);

  r = epotrs(CblasColMajor, 'U', n_rows, E->n,
	     B->v_vector, B->m, E->v_vector, E->m);
  assert(r == 0);

  if (LENKF_DEBUG) {
    fprintf(LENKF_DEBUG_FID, "\nB\\E:\n");
    full_c_rows_fprintf(LENKF_DEBUG_FID, E, 0, n_rows-1);
  }

  TOC(8);
}


static void compute_e(arg_bundle *ab, const sparse_rcs *H, int row_H,
		      const int n_rows, const elem *y, const int inc_y) {
  int row, index_H, row2;

  const vector *x_mean = ab->x_mean;
  vector *scratch  = ab->scratch;

  TIC(9);

  assert(H);
  assert(row_H >= 0 && row_H + n_rows - 1 < H->m);

  /* Otherwise scratch will not be large enough */
  assert(n_rows <= ab->config->N);

  ecopy(n_rows, y, inc_y, scratch->v, 1);

  if (LENKF_DEBUG) {
    fprintf(LENKF_DEBUG_FID, "\ny:\n");
    vector_fnprintf(LENKF_DEBUG_FID, scratch, n_rows);
  }

  for (row = row_H; row < row_H + n_rows; row++) {
    for (index_H = H->r[row]; index_H < H->r[row+1]; index_H++) {
      row2 = H->j[index_H];
      scratch->v[row - row_H] -= H->v[index_H] * x_mean->v[row2];
    }
  }

  if (LENKF_DEBUG) {
    fprintf(LENKF_DEBUG_FID, "\ne:\n");
    vector_fnprintf(LENKF_DEBUG_FID, scratch, n_rows);
  }

  TOC(9);
}


static void compute_B_rdiv_e(arg_bundle *ab, const int n_rows) {
  int r;

  const full_c *B = ab->B;
  const vector *scratch = ab->scratch;
  elem *e = ab->scratch->v;

  TIC(10);

  r = epotrs(CblasColMajor, 'U', n_rows, 1, B->v_vector,
	     B->n, e, n_rows);

  assert(r == 0);

  if (LENKF_DEBUG) {
    fprintf(LENKF_DEBUG_FID, "\nB\\e:\n");
    vector_fnprintf(LENKF_DEBUG_FID, scratch, n_rows);
  }

  TOC(10);
}


static void compute_X_update(arg_bundle *ab) {
  int i, j;

  const elem update_epsilon = ab->config->update_epsilon;
  sparse_lil *P_HT = ab->P_HT;
  const full_c *E = ab->E;
  vector *x_mean = ab->x_mean;

  full_r *X = ab->e->X;
  sparse_lil_row_it P_HT_it;
  elem P_HT_v;

  TIC(11);

  for (i = 0; i < P_HT->m; i++) {
    sparse_lil_row_it_init(P_HT, i, &P_HT_it);

    while (sparse_lil_row_it_has_next(&P_HT_it)) {
      P_HT_v = sparse_lil_row_it_next(&P_HT_it);
      j = P_HT_it.j;

      if ((update_epsilon == 0) ||
	  (fabs(P_HT_v/x_mean->v[i]) > update_epsilon)) {
	/* Matrix element indexed in [j][i] order because the E matrix
	   is stored column-wise */
	eaxpy(X->n, P_HT_v, &(E->v[0][j]), E->m, &(X->v[i][0]), 1);
      }
    }
  }

  if (LENKF_DEBUG) {
    fprintf(LENKF_DEBUG_FID, "\nX:\n");
    full_r_fprintf(LENKF_DEBUG_FID, X);
  }

  TOC(11);
}


static void compute_x_mean_update(arg_bundle *ab) {
  int i, j;

  const elem update_epsilon = ab->config->update_epsilon;
  sparse_lil *P_HT = ab->P_HT;
  vector *e = ab->scratch;
  vector *x_mean = ab->x_mean;

  sparse_lil_row_it P_HT_it;
  elem P_HT_v;

  TIC(12);


  for (i = 0; i < P_HT->m; i++) {
    sparse_lil_row_it_init(P_HT, i, &P_HT_it);

    while (sparse_lil_row_it_has_next(&P_HT_it)) {
      P_HT_v = sparse_lil_row_it_next(&P_HT_it);
      j = P_HT_it.j;

      if ((update_epsilon == 0) ||
	  (fabs(P_HT_v/x_mean->v[i]) > update_epsilon)) {
	x_mean->v[i] += P_HT_v * e->v[j];
      }
    }
  }

  if (LENKF_DEBUG) {
    fprintf(LENKF_DEBUG_FID, "\nx_mean:\n");
    vector_fprintf(LENKF_DEBUG_FID, x_mean);
  }

  TOC(12);
}



/* Compute the measurement update with localization */
static void compute_measurement_update(arg_bundle *ab,
				       const int i,
				       const sparse_rcs *H, int row_H,
				       const char name_H,
				       const elem *y, const int inc_y,
				       const elem *R_sqrt, const int inc_R_sqrt,
				       const int n_rows) {
  ensemble *e = ab->e;
  vector *x_mean = ab->x_mean;

  assert(i >= 0 && i < ab->config->T);
  assert(y);
  assert(R_sqrt);


  assert(row_H >= 0 && row_H < H->m);

  TIC_MU_TU(0);
  /* No longer subtracting mean at each measurment update */
  /*ensemble_mean(e, scratch);
    ensemble_subtract_mean(e, scratch);*/

  if (LENKF_DEBUG) {
    fprintf(LENKF_DEBUG_FID, "\n--------------------------------------------------------------------------------\n");
    fprintf(LENKF_DEBUG_FID, "i = %d\trow_%c = %d\tn_rows = %d\n",
	    i, name_H, row_H, n_rows);
    fprintf(LENKF_DEBUG_FID, "\nx_mean:\n");
    vector_fprintf(LENKF_DEBUG_FID, x_mean);
    fprintf(LENKF_DEBUG_FID, "\nX:\n");
    ensemble_fprintf(LENKF_DEBUG_FID, e);
    fprintf(LENKF_DEBUG_FID, "\n%c(%d:%d, :):\n", name_H,
	    row_H, row_H + n_rows - 1);
    sparse_rcs_rows_fprintf(LENKF_DEBUG_FID, H, row_H, row_H + n_rows - 1);
  }

  compute_P_HT(ab, H, row_H, name_H, n_rows);

  compute_H_P_HT(ab, H, row_H, name_H, n_rows);

  compute_H_P_HT_plus_R(ab, R_sqrt, inc_R_sqrt, name_H, n_rows);

  compute_V(ab, R_sqrt, inc_R_sqrt, n_rows);

  compute_V_minus_H_X(ab, H, row_H, name_H, n_rows);

  compute_chol_B(ab, n_rows);

  compute_B_rdiv_E(ab, n_rows);

  compute_e(ab, H, row_H, n_rows, y, inc_y);

  compute_B_rdiv_e(ab, n_rows);

  compute_X_update(ab);

  compute_x_mean_update(ab);

  TOC_MU_TU(0);
}


/* Compute ensemble trace */
static void compute_trace(arg_bundle *ab, int i) {
  const lenkf_config *config = ab->config;
  ensemble *e = ab->e;
  vector *trace = ab->trace;
  vector *scratch = ab->scratch;

  assert(i >= 0 && i < config->T);

  if (config->save_trace) {
      ensemble_mean(e, scratch);
      ensemble_subtract_mean(e, scratch);
      trace->v[i] = ensemble_trace(e);
      ensemble_add_mean(e, scratch);
  }

  if (LENKF_DEBUG) {
    fprintf(LENKF_DEBUG_FID, "\ntrace:\n");
    fprintf_elem_n(LENKF_DEBUG_FID, trace->v[i]);
  }
}


/* Compute the time update */
static void compute_time_update(arg_bundle *ab) {
  ensemble *e = ab->e;
  const full_r *X = e->X;
  const vector *x_mean = ab->x_mean;

  TIC_MU_TU(1);

  if (!ab->config->F_equal_I) {
    vector_copy(ab->scratch, ab->x_mean);
    sparse_rcs_mvm(ab->F, ab->scratch, ab->x_mean);
  }

  /* Time update */
  ensemble_add_state_noise_wrapper(ab);

  if (LENKF_DEBUG) {
    fprintf(LENKF_DEBUG_FID, "\nX:\n");
    full_r_fprintf(LENKF_DEBUG_FID, X);

    fprintf(LENKF_DEBUG_FID, "\nx_mean:\n");
    vector_fprintf(LENKF_DEBUG_FID, x_mean);
  }

  TOC_MU_TU(1);
}

/******************************************************************************/

double compute_time_per_iter(const struct timeval *start,
                             const struct timeval *current,
                             const int i) {
  struct timeval elapsed;

  elapsed.tv_sec = current->tv_sec - start->tv_sec;
  elapsed.tv_usec = current->tv_usec - start->tv_usec;

  while (elapsed.tv_usec < 0) {
    elapsed.tv_sec--;
    elapsed.tv_usec += 1e6;
  }

  while (elapsed.tv_usec >= 1e6) {
    elapsed.tv_sec++;
    elapsed.tv_usec -= 1e6;
  }

  return (elapsed.tv_sec + elapsed.tv_usec/1e6)/i;
}

double time_elapsed(const struct timeval *start,
                    const struct timeval *current) {
  struct timeval elapsed;

  assert(start);
  assert(current);

  elapsed.tv_sec = current->tv_sec - start->tv_sec;
  elapsed.tv_usec = current->tv_usec - start->tv_usec;

  while (elapsed.tv_usec < 0) {
    elapsed.tv_sec--;
    elapsed.tv_usec += 1e6;
  }

  while (elapsed.tv_usec >= 1e6) {
    elapsed.tv_sec++;
    elapsed.tv_usec -= 1e6;
  }

  return (elapsed.tv_sec + elapsed.tv_usec/1e6);
}

void printf_time_info(const struct timeval *start_time,
		      unsigned int *buffer_strlen,
		      const int i, const int N_steps,
		      FILE *stamp_fid) {
  char buffer[BUFFER_SIZE], out_buffer[BUFFER_SIZE];
  struct timeval current_time;
  double time_per_iter;
  unsigned int l;

  sprintf(buffer, "%i of %i", i, N_steps-1);
  sprintf(out_buffer, "%-16s", buffer);
  if (i > 0) {
    gettimeofday(&current_time, NULL);
    time_per_iter = compute_time_per_iter(start_time, &current_time, i);
    sprintf_hms(buffer, time_per_iter * (N_steps - i));
    strcat(out_buffer, buffer);
    fprintf(stamp_fid, "(%s", buffer);
    strcat(out_buffer, "    ");
    fprintf(stamp_fid, "%s", "    ");
    sprintf_hms(buffer, time_elapsed(start_time, &current_time) +
		time_per_iter * (N_steps - i));
    fprintf(stamp_fid, "%s)\n", buffer);
    fflush(stamp_fid);
    strcat(out_buffer, buffer);
  }

  if (!LENKF_DEBUG) {
    if (*buffer_strlen > 0) {
      for (l = 0; l < *buffer_strlen; l++) {
	putchar('\b');
      }
    }
    if (i > 0 && *buffer_strlen > strlen(out_buffer)) {
      for (l = 0; l < *buffer_strlen - strlen(out_buffer); l++) {
	strcat(out_buffer, " ");
      }
    }
    fputs(out_buffer, stdout);
    fflush(stdout);
    *buffer_strlen = strlen(out_buffer) + 1;
  }
}

static void TIC(const int clock_index) {
  if (ENABLE_PROFILING) {
    assert(clock_index >= 0 && clock_index < N_CLOCKS);
    multi_sw_start(SW, clock_index);
  }
}

static void TOC(const int clock_index) {
  if (ENABLE_PROFILING) {
    assert(clock_index >= 0 && clock_index < N_CLOCKS);
    multi_sw_stop(SW, clock_index);
  }
}

static void TIC_MU_TU(const int clock_index) {
  if (ENABLE_PROFILING) {
    assert(clock_index >= 0 && clock_index < 2);
    multi_sw_start(SW_MU_TU, clock_index);
  }
}

static void TOC_MU_TU(const int clock_index) {
  if (ENABLE_PROFILING) {
    assert(clock_index >= 0 && clock_index < 2);
    multi_sw_stop(SW_MU_TU, clock_index);
  }
}


/******************************************************************************/

void ensemble_add_state_noise_wrapper(arg_bundle *ab) {
  assert(ab);

  if (ab->config->randn_conv) {
    if (ab->Q_sqrt.filter->N_phy[0] == 0) {
      return;
    }
  }
  else {
    if (ab->Q_sqrt.rcs->m == 0) {
      return;
    }
  }

  if (ab->config->randn_conv) {
    ensemble_add_noise_conv_ab(ab, ab->Q_sqrt.filter);
  }
  else {
    ensemble_add_noise_ab(ab, ab->Q_sqrt.rcs);
  }
}

void ensemble_init_wrapper(arg_bundle *ab) {
  assert(ab);

  if (ab->config->randn_conv) {
    ensemble_init_conv(ab);
  }
  else {
    ensemble_init(ab);
  }
}

/******************************************************************************/

void build_filename(char *buffer, int buffer_size, lenkf_config *config, char *prefix, int i) {
  snprintf(buffer, buffer_size, "%s/%s_%d", config->dir, prefix, i);
}

/*****************************************************************************/

void lenkf(full_c *x_mean_final, lenkf_config *config,
	   lenkf_stats *stats) {
  printf("LENKFLENKF");
  arg_bundle *ab;
  int i, row, n_rows;
  elem zero = 0, inv_sqrt_lambda = ((elem) 1) / sqrte(config->lambda);

  struct timeval start_time, current_time;
  time_t current_time_time_t;
  unsigned int buffer_strlen;

  char *user;
  char buffer[2*BUFFER_SIZE];

  FILE *stamp_fid;

  assert(config);
  assert(x_mean_final);
  assert(x_mean_final->n == config->T);
  assert(x_mean_final->m == config->N);

  LENKF_DEBUG = config->lenkf_debug;
  if (LENKF_DEBUG) {
    strcat(strncpy(buffer, config->dir, BUFFER_SIZE), "/");
    strncat(buffer, "lenkf.debug", BUFFER_SIZE);

    LENKF_DEBUG_FID = fopen(buffer, "w");
    assert(LENKF_DEBUG_FID);
  }

  user = getenv("USER");
  assert(user);

  strcat(strncpy(buffer, config->dir, BUFFER_SIZE), "/");
  strncat(buffer, "lenkf.log", BUFFER_SIZE);

  stamp_fid = fopen(buffer, "w");
  assert(stamp_fid);

  gettimeofday(&start_time, NULL);

  if (config->enable_profiling) {
    ENABLE_PROFILING = cfg_true;
    N_CLOCKS = 13;
    SW = multi_sw_create(N_CLOCKS);
    multi_sw_set_name(SW, 0,  "Init");
    multi_sw_set_name(SW, 1,  "P_HT");
    multi_sw_set_name(SW, 2,  "table");
    multi_sw_set_name(SW, 3,  "H_P_HT");
    multi_sw_set_name(SW, 4,  "H_P_HT+R");
    multi_sw_set_name(SW, 5,  "V");
    multi_sw_set_name(SW, 6,  "V-H_X");
    multi_sw_set_name(SW, 7,  "chol(B)");
    multi_sw_set_name(SW, 8,  "B\\E");
    multi_sw_set_name(SW, 9,  "e");
    multi_sw_set_name(SW, 10, "B\\e");
    multi_sw_set_name(SW, 11, "X update");
    multi_sw_set_name(SW, 12, "mean update");

    SW_MU_TU = multi_sw_create(2);
    multi_sw_set_name(SW_MU_TU, 0, "MU");
    multi_sw_set_name(SW_MU_TU, 1, "TU");
  }

  /* The final step of arg_bundle_create is arg_bundle_check.  The
     constructed arg_bundle should be consistent - i.e., all data
     structures will have the correct dimensions. */
  ab = arg_bundle_create(config);
  /* ASSUMPTION: All members of ab are of the proper dimension.  This
     is verified during the construction. */

  TIC(0);
  ensemble_init_wrapper(ab);
  TOC(0);

  if (config->save_intermediate) {
    build_filename(buffer, BUFFER_SIZE, config, "x_hat_prior", 0);
    vector_export(buffer, ab->x_mean);
    build_filename(buffer, BUFFER_SIZE, config, "X_prior", 0);
    ensemble_export(buffer, ab->e);
  }

  if (LENKF_DEBUG) {
    fputs("Initial ensemble:\n", LENKF_DEBUG_FID);
    ensemble_fprintf(LENKF_DEBUG_FID, ab->e);
  }

  buffer_strlen = 0;

  for (i = 0; i < config->T; i++) {
    time(&current_time_time_t);
    fprintf(stamp_fid, "i = %d  (I = %d)  \t%s", i, config->T, ctime(&current_time_time_t));
    fflush(stamp_fid);

    if (!config->quiet_mode && config->T > 1) {
      printf_time_info(&start_time, &buffer_strlen, i, config->T, stamp_fid);
    }

    /* MEASUREMENT UPDATE */
    arg_bundle_time_step(ab, i);

    for (row = 0; row < config->M; row += config->M_block) {
      if (row + config->M_block > config->M) {
	n_rows = config->M - row;
      }
      else {
	n_rows = config->M_block;
      }

      if (config->poisson_noise) {
	vector *R_sqrt_i;
	int n;

	R_sqrt_i = vector_create(n_rows);

	for (n = 0; n < n_rows; n++) {
	  R_sqrt_i->v[n] = sqrte(ab->y->v[row + n]);
	  if (R_sqrt_i->v[n] < config->poisson_eps) {
	    R_sqrt_i->v[n] = config->poisson_eps;
	  }
	}

	compute_measurement_update(ab, i, ab->H, row, 'H',
				   &(ab->y->v[row]), 1,
				   R_sqrt_i->v, 1,
				   n_rows);

	vector_destroy(&R_sqrt_i);
      }
      else {
	compute_measurement_update(ab, i, ab->H, row, 'H',
				   &(ab->y->v[row]), 1,
				   &(ab->R_sqrt->v->v[row]), 1,
				   n_rows);
      }
    }


    if (config->regularize) {
      for (row = 0; row < ab->D->m; row += config->M_block) {
	if (row + config->M_block > ab->D->m) {
	  n_rows = ab->D->m - row;
	}
	else {
	  n_rows = config->M_block;
	}

	compute_measurement_update(ab, i, ab->D, row, 'D',
				   &zero, 0, &inv_sqrt_lambda, 0,
				   n_rows);
      }
    }

    /* [col][row] indexing because x_mean_final is stored column-wise */
    ecopy(config->N, ab->x_mean->v, 1, &(x_mean_final->v[i][0]), 1);

    compute_trace(ab, i);

    if (config->save_intermediate) {
      build_filename(buffer, BUFFER_SIZE, config, "x_hat_posterior", i);
      vector_export(buffer, ab->x_mean);
      build_filename(buffer, BUFFER_SIZE, config, "X_posterior", i);
      ensemble_export(buffer, ab->e);
    }

    /* TIME UPDATE */
    compute_time_update(ab);

    if (config->save_intermediate) {
      build_filename(buffer, BUFFER_SIZE, config, "x_hat_prior", i+1);
      vector_export(buffer, ab->x_mean);
      build_filename(buffer, BUFFER_SIZE, config, "X_prior", i+1);
      ensemble_export(buffer, ab->e);
    }
  }

  stats->res = get_res();

  if (config->save_trace) {
    vector_export(config->trace_filename, ab->trace);
  }

  /* CLEANUP */
  arg_bundle_destroy(&ab);

  if (config->randn_conv) {
    fftwe_cleanup();
  }

  if (config->enable_profiling) {
    printf("\n");
    multi_sw_printf(SW);
    multi_sw_printf_total(SW);

    printf("\n");
    multi_sw_printf(SW_MU_TU);
    multi_sw_printf_total(SW_MU_TU);

    multi_sw_destroy(&SW);
    multi_sw_destroy(&SW_MU_TU);
  }

  gettimeofday(&current_time, NULL);
  stats->exec_time = time_elapsed(&start_time, &current_time);

  fclose(stamp_fid);

  if (LENKF_DEBUG) {
    fclose(LENKF_DEBUG_FID);
  }

  return;
}
