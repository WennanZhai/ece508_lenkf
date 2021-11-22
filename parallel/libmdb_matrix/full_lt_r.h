#ifndef FULL_LT_R_H
#define FULL_LT_R_H

#include "elem.h"
#include "full_r.h"


typedef struct {
  int m, n;
  elem *v;
  int N;
} lt_r;


lt_r *lt_r_create(int m, int n);
void lt_r_destroy(lt_r **A);
lt_r *lt_r_import(const char *filename);
elem lt_r_get(const lt_r *A, const int i, const int j);
elem lt_r_get_submatrix(const lt_r *A, const int i, const int j,
			const int M, const int N);
void lt_r_set(const lt_r *A, const int i, const int j, const elem v);
void lt_r_add(const lt_r *A, const int i, const int j, const elem v);
void lt_r_printf(const lt_r *A);
void lt_r_fprintf(FILE *fid, const lt_r *A);
void lt_r_submatrix_fprintf(FILE *fid, const lt_r *A, const int M, const int N);
void lt_r_set0(lt_r *A);

#endif
