#include <stdlib.h>
#include <assert.h>

#include "arg_bundle.h"
#include "randn.h"

#define BUFFER_SIZE 256


static void open_files(arg_bundle *ab);


arg_bundle *arg_bundle_create(lenkf_config *config) {
  arg_bundle *ab;
  
  assert(config);
  
  ab = malloc(sizeof(arg_bundle));
  assert(ab);

  ab->config = config;
  
  ab->e = ensemble_create(config->N, config->L);

  ab->P_HT = NULL;

  ab->E = full_c_create(config->M_block, config->L);
  ab->B = full_c_create(config->M_block, config->M_block);
  
  ab->scratch = vector_create(config->N);

  open_files(ab);

  if (config->randn_conv) {
    ab->u_conv = r_filter_new_create_same_dim(ab->PI_sqrt.filter);
  }
  else {
    ab->u_conv = NULL;
  }

  ab->trace = ab->config->save_trace ? vector_create(ab->config->T) : NULL;

  ab->y = NULL;
  ab->H = NULL;
  ab->R_sqrt = NULL;
  
  ab->i = -1;

  arg_bundle_check(ab);
  
  return ab;
}

void arg_bundle_destroy(arg_bundle **ab) {
  assert(ab);
  assert(*ab);

  arg_bundle_check(*ab);

  ensemble_destroy(&(*ab)->e);

  vector_destroy(&(*ab)->x_mean);

  if ((*ab)->P_HT) {
    sparse_lil_destroy(&(*ab)->P_HT);
  }

  full_c_destroy(&(*ab)->E);
  full_c_destroy(&(*ab)->B);
  
  vector_destroy(&(*ab)->scratch);

  vector_destroy(&(*ab)->y);
  sparse_rcs_destroy(&(*ab)->D);
  
  if (!(*ab)->config->poisson_noise) {
    diag_destroy(&(*ab)->R_sqrt);
  }
  
  sparse_rcs_destroy(&(*ab)->H);
  
  sb_toe_r_destroy(&(*ab)->C);
  sb_toe_r_nz_it_destroy(&(*ab)->C_it);
  
  if ((*ab)->config->randn_conv) {
    r_filter_new_destroy(&(*ab)->PI_sqrt.filter);
    r_filter_new_destroy(&(*ab)->Q_sqrt.filter);
    r_filter_new_destroy(&(*ab)->u_conv);
  }
  else {
    sparse_rcs_destroy(&(*ab)->PI_sqrt.rcs);
    sparse_rcs_destroy(&(*ab)->Q_sqrt.rcs);
  }

  if (!(*ab)->config->F_equal_I) {
    sparse_rcs_destroy(&(*ab)->F);
  }

  if ((*ab)->config->save_trace) {
    vector_destroy(&(*ab)->trace);
  }
  
  free(*ab);
  *ab = NULL;
}

void arg_bundle_check(arg_bundle *ab) {
  int i;
  
  assert(ab);

  assert(ab->config != NULL);
  assert(ab->e != NULL);
  assert(ab->x_mean != NULL);

  assert(ab->e->N == ab->config->N);
  assert(ab->e->L == ab->config->L);

  assert(ab->x_mean->n == ab->config->N);

  assert(ab->E != NULL);
  assert(ab->B != NULL);

  assert(ab->E->m == ab->config->M_block);
  assert(ab->E->n == ab->config->L);
  
  assert(ab->B->m == ab->B->n);
  assert(ab->B->m == ab->config->M_block);

  assert(ab->C->dim->N[0] == ab->config->N);

  if (ab->D->m > 0) {
    assert(ab->D->n == ab->config->N);
  }
  else {
    assert(ab->D->n == 0);
  }

  if (ab->config->save_trace) {
    assert(ab->trace->n == ab->config->T);
  }
  else {
    assert(ab->trace == NULL);
  }

  if (ab->config->F_equal_I) {
    assert(ab->F == NULL);
  }
  else {
    assert(ab->F->n == ab->config->N);
    assert(ab->F->m == ab->F->n);
  }
  
  assert(ab->x_mean->n == ab->config->N);
  assert(ab->scratch->n == ab->config->N);

  if (ab->config->randn_conv) {
    assert(ab->PI_sqrt.filter->rank == ab->config->rank);
    assert(ab->Q_sqrt.filter->rank == ab->config->rank);
    
    for (i = 0; i < ab->config->rank; i++) {
      assert(ab->PI_sqrt.filter->n_log[i] >= ab->config->n[i]);
      assert(ab->Q_sqrt.filter->n_log[i] >= ab->config->n[i]);
    }
  }
  else {
    assert(ab->PI_sqrt.rcs->n == ab->PI_sqrt.rcs->m);
    assert(ab->PI_sqrt.rcs->n == ab->config->N);

    assert(ab->Q_sqrt.rcs->n == ab->Q_sqrt.rcs->m);
    assert(ab->Q_sqrt.rcs->n == ab->config->N);
  }
}

void open_files(arg_bundle *ab) {
  int N, M, T;

  N = ab->config->N;
  M = ab->config->M;
  T = ab->config->T;

  ab->D = sparse_rcs_import(ab->config->D_filename);

  ab->C = sb_toe_r_import(ab->config->C_filename);
  ab->C_it = sb_toe_r_nz_it_create(ab->C);

  ab->x_mean = vector_import(ab->config->x0_filename);

  if (ab->config->randn_conv) {
    ab->PI_sqrt.filter = r_filter_new_import(ab->config->PI_sqrt_filename);
    r_filter_new_scal(ab->PI_sqrt.filter, ab->config->alpha_PI_sqrt);
    r_filter_new_dft(ab->PI_sqrt.filter);

    ab->Q_sqrt.filter = r_filter_new_import(ab->config->Q_sqrt_filename);
    r_filter_new_dft(ab->Q_sqrt.filter);
  }
  else {
    ab->PI_sqrt.rcs = sparse_rcs_import(ab->config->PI_sqrt_filename);
    sparse_rcs_scal(ab->PI_sqrt.rcs, ab->config->alpha_PI_sqrt);
    ab->Q_sqrt.rcs = sparse_rcs_import(ab->config->Q_sqrt_filename);
  }

  ab->F = ab->config->F_equal_I ? NULL :
    sparse_rcs_import(ab->config->F_filename);
}


void arg_bundle_time_step(arg_bundle *ab, const int i) {
  assert(ab);

  assert(i >= 0 && i < ab->config->T);
  ab->i = i;

  if (ab->i == 0) {
    assert(ab->y == NULL);
    assert(ab->H == NULL);
    assert(ab->R_sqrt == NULL);
  }
  else {
    vector_destroy(&ab->y);
    sparse_rcs_destroy(&ab->H);
    if (!ab->config->poisson_noise) {
      diag_destroy(&ab->R_sqrt);
    }
    else {
      assert(ab->R_sqrt == NULL);
    }
  }

  ab->y = vector_import(ab->config->y_filename_list[i]);
  ab->H = sparse_rcs_import(ab->config->H_filename_list[i]);
  ab->R_sqrt = diag_import(ab->config->R_sqrt_filename_list[i]);
  
  assert(ab->y->n == ab->H->m);
  assert(ab->H->n == ab->config->N);

  if (!ab->config->poisson_noise) {
    assert(ab->H->m == ab->R_sqrt->m);
    assert(ab->R_sqrt->m == ab->R_sqrt->n);
  }
  
  ab->config->M = ab->H->m;
}
