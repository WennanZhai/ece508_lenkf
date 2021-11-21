#ifndef FULL_R_H
#define FULL_R_H

#include <stdio.h>

#include "elem.h"


typedef struct {
  int m, n;
  elem *v_vector;
  elem **v;
} full_r;


full_r *full_r_create(int m, int n);
void full_r_destroy(full_r **A);
void full_r_set(full_r *A, int i, int j, elem v);
elem full_r_get(const full_r *A, int i, int j);
void full_r_r_foreach(const full_r *A, int i, void (*func)(elem));
void full_r_foreach(const full_r *A, void (*func)(elem));
void full_r_printf(const full_r *A);
void full_r_fprintf(FILE *fid, const full_r *A);
void full_r_rows_fprintf(FILE *fid, const full_r *A,
			 const int i1, const int i2);
void full_r_submatrix_fprintf(FILE *fid, const full_r *A,
			      const int M, const int N);
full_r *full_r_import(const char *filename);
void full_r_export(const char *filename, const full_r *A);

elem *full_r_get_ptr(const full_r *A, int i, int j);
void full_r_set0(full_r *A);
void full_r_setI(full_r *A);
void full_r_copy(full_r *Y, const full_r *X);
int full_r_check_index(const full_r *A, int i, int j);

void full_r_scal(full_r *A, const elem alpha);
elem full_r_trace(const full_r *A);
int full_r_nnz(const full_r *A);
void full_r_mmm(const full_r *A, const full_r *B, const elem alpha, full_r *C);
void full_r_mmmT(const full_r *A, const full_r *B, const elem alpha, full_r *C);

#endif
