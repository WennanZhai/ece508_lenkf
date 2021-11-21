#ifndef ZFULL_C_H
#define ZFULL_C_H

#include "elem.h"

typedef struct {
  int m, n;
  z_elem *v_vector;
  z_elem **v;
} zfull_c;

zfull_c *zfull_c_create(int m, int n);
void zfull_c_destroy(zfull_c **A);
void zfull_c_printf(const zfull_c *A);
void zfull_c_export(const char *filename, const zfull_c *A);
void zfull_c_set0(zfull_c *A);

#endif
