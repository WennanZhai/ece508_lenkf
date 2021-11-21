#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "lenkf_config.h"
#include "randn.h"


extern cfg_bool_t RANDN_DEBUG;


static char **load_file_list(const char *file_list_filename, int T);


lenkf_config *lenkf_config_init(void) {
  lenkf_config *config;

  config = malloc(sizeof(lenkf_config));
  assert(config);

  config->dir = NULL;

  config->x_hat_filename = NULL;
  config->trace_filename = NULL;

  config->x0_filename = NULL;
  config->PI_sqrt_filename = NULL;
  config->alpha_PI_sqrt = 1;

  config->y_list_filename = NULL;
  config->H_list_filename = NULL;
  config->R_sqrt_list_filename = NULL;

  config->y_filename_list = NULL;
  config->H_filename_list = NULL;
  config->R_sqrt_filename_list = NULL;

  config->D_filename = NULL;
  config->lambda = 0;

  config->C_filename = NULL;

  config->F_filename = NULL;
  config->Q_sqrt_filename = NULL;

  config->rank = 0;
  config->n = NULL;

  config->N = 0;
  config->M = 0;
  config->M_block = 0;
  config->T = 0;
  config->L = 0;

  config->seed = -1;

  config->update_epsilon = 0;

  config->poisson_noise = cfg_false;
  config->poisson_eps = 1;

  config->regularize = cfg_false;
  config->F_equal_I = cfg_false;
  config->randn_conv = cfg_false;

  config->quiet_mode = cfg_false;
  config->save_trace = cfg_false;
  config->save_intermediate = cfg_false;
  config->enable_profiling = cfg_false;
  config->lenkf_debug = cfg_false;

  return config;
}

void lenkf_config_destroy(lenkf_config **config) {
  int i;

  assert(config);
  assert(*config);

  free((*config)->dir);

  free((*config)->x_hat_filename);
  free((*config)->trace_filename);

  free((*config)->y_list_filename);
  free((*config)->H_list_filename);
  free((*config)->R_sqrt_list_filename);

  for (i=0; i < (*config)->T; i++) {
    free((*config)->y_filename_list[i]);
    free((*config)->H_filename_list[i]);
    free((*config)->R_sqrt_filename_list[i]);
  }

  free((*config)->y_filename_list);
  free((*config)->H_filename_list);
  free((*config)->R_sqrt_filename_list);

  free((*config)->D_filename);
  free((*config)->C_filename);

  free((*config)->n);

  if ((*config)->F_equal_I) {
    assert((*config)->F_filename == NULL);
  }
  else {
    free((*config)->F_filename);
  }

  free((*config)->Q_sqrt_filename);

  free((*config)->x0_filename);
  free((*config)->PI_sqrt_filename);

  free(*config);
  *config = NULL;
}


void lenkf_config_check(cfg_t *cfg, const lenkf_config *config) {
  int i, N;

  assert(cfg);
  assert(config);

  assert(config->dir);

  assert(config->x_hat_filename);
  assert((config->trace_filename == NULL) != config->save_trace);

  assert(config->x0_filename);
  assert(config->PI_sqrt_filename);
  assert(config->alpha_PI_sqrt > 0);

  assert(config->y_list_filename[0] != '\0');
  assert(config->H_list_filename[0] != '\0');
  assert(config->R_sqrt_list_filename[0] != '\0');

  assert(config->y_filename_list);
  assert(config->H_filename_list);
  assert(config->R_sqrt_filename_list);
  assert(config->D_filename);
  assert(config->C_filename);

  assert((config->F_filename == NULL) == config->F_equal_I);
  assert(config->Q_sqrt_filename);

  assert(config->rank > 0);

  N = 1;
  for (i = 0; i < config->rank; i++) {
    N *= config->n[i];
  }

  assert(config->N == N);

  assert(config->N > 0);
  assert(config->M_block > 0 && config->M_block <= config->N);
  assert(config->T >= 0);
  assert(config->L > 0);

  assert(config->seed == -1 || config->seed >= 1);

  assert(config->update_epsilon >= 0);

  if (config->regularize) {
    assert(config->lambda >= 0);
  }
  else {
    assert(config->lambda == 0);
  }

  if (config->poisson_noise) {
    assert(config->poisson_eps >= 0);
  }
  else {
    assert(config->poisson_eps == 1);
  }
}


void lenkf_config_get_options(cfg_t *cfg, lenkf_config *config) {
  int i;

  assert(cfg);
  assert(config);

  config->rank = cfg_getint(cfg, "rank");
  assert(config->rank > 0);

  assert(cfg_size(cfg, "n") == config->rank);
  config->n = malloc(config->rank * sizeof(int));
  assert(config->n);

  for (i = 0; i < config->rank; i++) {
    config->n[i] = cfg_getnint(cfg, "n", i);
  }

  config->N = cfg_getint(cfg, "N");

  config->M_block = cfg_getint(cfg, "M_block");
  if (config->M_block == 0) {
    config->M_block = config->M;
  }
  config->T = cfg_getint(cfg, "T");
  config->L = cfg_getint(cfg, "L");

  config->seed = cfg_getint(cfg, "seed");

  config->update_epsilon = cfg_getfloat(cfg, "update_epsilon");

  config->lambda = cfg_getfloat(cfg, "lambda");

  config->poisson_noise = cfg_getbool(cfg, "poisson_noise");
  config->poisson_eps = cfg_getfloat(cfg, "poisson_eps");

  config->regularize = cfg_getbool(cfg, "regularize");
  config->F_equal_I = cfg_getbool(cfg, "F_equal_I");
  config->randn_conv = cfg_getbool(cfg, "randn_conv");

  config->quiet_mode = cfg_getbool(cfg, "quiet_mode");
  config->save_trace = cfg_getbool(cfg, "save_trace");
  config->save_intermediate = cfg_getbool(cfg, "save_intermediate");
  config->enable_profiling = cfg_getbool(cfg, "enable_profiling");
  config->lenkf_debug = cfg_getbool(cfg, "lenkf_debug");

  config->dir = strdup(cfg_getstr(cfg, "dir"));

  config->x_hat_filename = strdup(cfg_getstr(cfg, "x_hat_filename"));
  config->trace_filename = strdup(cfg_getstr(cfg, "trace_filename"));

  config->x0_filename = strdup(cfg_getstr(cfg, "x0_filename"));
  config->PI_sqrt_filename = strdup(cfg_getstr(cfg, "PI_sqrt_filename"));
  config->alpha_PI_sqrt = cfg_getfloat(cfg, "alpha_PI_sqrt");

  config->y_list_filename = strdup(cfg_getstr(cfg, "y_list_filename"));
  config->H_list_filename = strdup(cfg_getstr(cfg, "H_list_filename"));
  config->R_sqrt_list_filename = strdup(cfg_getstr(cfg, "R_sqrt_list_filename"));

  config->y_filename_list = load_file_list(config->y_list_filename, config->T);
  config->H_filename_list = load_file_list(config->H_list_filename, config->T);
  config->R_sqrt_filename_list = load_file_list(config->R_sqrt_list_filename, config->T);

  config->C_filename = strdup(cfg_getstr(cfg, "C_filename"));
  config->D_filename = strdup(cfg_getstr(cfg, "D_filename"));

  if (!config->F_equal_I) {
    config->F_filename = strdup(cfg_getstr(cfg, "F_filename"));
  }
  config->Q_sqrt_filename = strdup(cfg_getstr(cfg, "Q_sqrt_filename"));

  RANDN_DEBUG = cfg_getbool(cfg, "randn_debug");
  if (RANDN_DEBUG) {
    randn_set_dir(config->dir);
  }
}


static char **load_file_list(const char *file_list_filename, int T) {
  FILE *fid;
  char **file_list;
  const int BUFFER_SIZE = 1024;
  char buffer[BUFFER_SIZE];
  int i;
  char *result;
  int result_strlen;

  assert(file_list_filename);
  assert(T >= 0);

  file_list = malloc(T * sizeof(char *));
  assert(file_list);

  if (file_list_filename[0] != '\0') {
      fid = fopen(file_list_filename, "r");
      assert(fid);

      for (i = 0; i < T; i++) {
          result = fgets(buffer, BUFFER_SIZE, fid);
          assert(result);

          file_list[i] = strdup(result);

          result_strlen = strlen(result);
          assert(file_list[i][result_strlen - 1] == '\n');
          file_list[i][result_strlen - 1] = '\0';
      }

      fclose(fid);
  }

  return file_list;
}
