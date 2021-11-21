#ifndef LENKF_CONFIG_H
#define LENKF_CONFIG_H

#include <confuse.h>

#ifdef LENKF_FLOAT_ELEM
#include "mdb_matrix_s.h"
#elif defined LENKF_DOUBLE_ELEM
#include "mdb_matrix_d.h"
#else
#error ?
#endif

#include "util.h"


typedef struct {
  char *dir;

  int rank;
  int *n;

  int N, M, M_block, T, L;

  long int seed;

  char *x_hat_filename;
  char *trace_filename;

  char *x0_filename;
  char *PI_sqrt_filename;
  elem alpha_PI_sqrt;

  char *y_list_filename;
  char *H_list_filename;
  char *R_sqrt_list_filename;

  char **y_filename_list;
  char **H_filename_list;
  char **R_sqrt_filename_list;

  char *D_filename;
  elem lambda;

  char *C_filename;

  char *F_filename;
  char *Q_sqrt_filename;

  elem update_epsilon;

  cfg_bool_t poisson_noise;
  elem poisson_eps;

  cfg_bool_t regularize;
  cfg_bool_t F_equal_I;
  cfg_bool_t randn_conv;

  cfg_bool_t quiet_mode;
  cfg_bool_t save_trace;
  cfg_bool_t save_intermediate;
  cfg_bool_t enable_profiling;
  cfg_bool_t lenkf_debug;
} lenkf_config;

lenkf_config *lenkf_config_init(void);
void lenkf_config_destroy(lenkf_config **config);
void lenkf_config_get_options(cfg_t *cfg, lenkf_config *config);
void lenkf_config_check(cfg_t *cfg, const lenkf_config *config);

#endif
