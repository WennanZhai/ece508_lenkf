#ifndef ARG_BUNDLE_H
#define ARG_BUNDLE_H

#ifdef LENKF_FLOAT_ELEM
#include "mdb_matrix_s.h"
#elif defined LENKF_DOUBLE_ELEM
#include "mdb_matrix_d.h"
#else
#error ?
#endif

#include "lenkf_config.h"


typedef union {
  sparse_rcs   *rcs;
  r_filter_new *filter;
} noise_cov;

typedef struct arg_bundle_struct {
  lenkf_config *config;

  /* A little forward reference trickery is necessary. */
  struct ensemble_struct *e;

  /* Length N vectors */
  vector *x_mean;
  vector *scratch;

  sparse_lil *P_HT;
   
  /* A M_block x L matrix */
  full_c *E;

  /* A M_block x M_block  matrix */
  full_c *B;
  
  r_filter_new *u_conv;
  
  /* Length T vector */
  vector *trace;

  vector *y;
  sparse_rcs *H;
  diag *R_sqrt;
  int i;
  
  sparse_rcs *F;
  
  sparse_rcs *D;
  
  noise_cov PI_sqrt;
  noise_cov Q_sqrt;

  sb_toe_r *C;
  sb_toe_r_it *C_it;
} arg_bundle;

/* Slight trickery to satisfy compiler.  Note that arg_bundle is used
   in ensemble.h.  Thus, we must define arg_bundle prior to #including
   the file.  Using a forward reference to struct arg_bundle_struct in
   the prototype for ensemble_init_conv_2D, for example, caused the
   compiler to generate a warning informing that this is probably not
   what we wanted to do (though, this is perfectly OK). */
#include "ensemble.h"

arg_bundle *arg_bundle_create(lenkf_config *config);
void arg_bundle_destroy(arg_bundle **ab);
void arg_bundle_check(arg_bundle *ab);
void arg_bundle_time_step(arg_bundle *ab, const int i);

#endif
