#ifndef FULL_C_H
#define FULL_C_H

#include <stdio.h>

#include "elem.h"
#include "sparse_ccs.h"


typedef struct {
  int m, n;
  elem *v_vector;
  elem **v;
} full_c;


full_c *full_c_create(int m, int n);
void full_c_destroy(full_c **A);
full_c *full_c_import(const char *filename);
void full_c_export(const char *filename, const full_c *A);
void full_c_export_partial(const char *filename, const full_c *A, const int n);
void full_c_set(full_c *A, int i, int j, elem v);
void full_c_set0(full_c *A);
elem full_c_get(const full_c *A, int i, int j);
void full_c_r_foreach(const full_c *A, int i, void (*func)(elem));
void full_c_foreach(const full_c *A, void (*func)(elem));
void full_c_fprintf(FILE *fid, const full_c *A);
void full_c_submatrix_fprintf(FILE *fid, const full_c *A,
			      const int M, const int N);
void full_c_cols_fprintf(FILE *fid, const full_c *A,
			 const int col1, const int col2);
void full_c_rows_fprintf(FILE *fid, const full_c *A,
			 const int row1, const int row2);
void full_c_printf(const full_c *A);
void full_c_export_raw(const full_c *A, const char *filename);
void full_c_get_submatrix(full_c *A, const full_c *B, int i, int j, int m, int n);
void full_c_copy(full_c *A, const full_c *B);
void full_c_mmult_ut(full_c *A, const full_c *B);
void full_c_get_submatrix_ccs(full_c *A, const sparse_ccs *B, int i, int j, int m, int n);
void full_c_scal(full_c *A, const elem alpha);
elem full_c_trace(const full_c *A);
int full_c_nnz(const full_c *A);
int full_c_rows_nnz(const full_c *A, const int row1, const int row2);
void full_c_mmm(const full_c *A, const full_c *B, const elem alpha, full_c *C);
void full_c_mmmT(const full_c *A, const full_c *B, const elem alpha, full_c *C);


#endif
