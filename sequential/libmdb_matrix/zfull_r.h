#ifndef ZFULL_R_H
#define ZFULL_R_H

#include "elem.h"

typedef struct {
  int m, n;
  z_elem *v_vector;
  z_elem **v;
} zfull_r;

zfull_r *zfull_r_create(int m, int n);
void zfull_r_destroy(zfull_r **A);
void zfull_r_mmm(const zfull_r *A, const zfull_r *B, zfull_r *C);
void zfull_r_printf(const zfull_r *A);
void zfull_r_export(const char *filename, const zfull_r *A);
void zfull_r_set0(zfull_r *A);
void zfull_r_r_foreach(const zfull_r *A, int i, void (*func)(const z_elem *));

#endif
