#ifndef LENKF_H
#define LENKF_H

#ifdef LENKF_FLOAT_ELEM
#include "mdb_matrix_s.h"
#elif defined LENKF_DOUBLE_ELEM
#include "mdb_matrix_d.h"
#else
#error ?
#endif

#include "lenkf_config.h"

typedef struct {
  double exec_time;
  double res;
} lenkf_stats;

void lenkf(full_c *x_mean_final, lenkf_config *config,
	   lenkf_stats *stats);

#endif
