#ifndef FULL_UT_C_H
#define FULL_UT_C_H

#include "full_c.h"


typedef struct {
  int m, n;
  elem *v;
  int N;
} ut_c;


ut_c *ut_c_create(int m, int n);
void ut_c_destroy(ut_c **A);
ut_c *ut_c_import(const char *filename);
elem ut_c_get(const ut_c *A, int i, int j);
void ut_c_set(ut_c *A, int i, int j, elem v);
void ut_c_foreach(const ut_c *A, void (*func)(elem));
void ut_c_printf(const ut_c *A);
void ut_c_r_foreach(const ut_c *A, int i, void (*func)(elem));
void ut_c_r_printf(const ut_c *A, int i);
void full_c_2_ut_c(const full_c *A, ut_c *B);
void full_c_times_ut_c(full_c *A, const ut_c *B);

#endif
