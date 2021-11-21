#ifndef SPARSE_RCS_H
#define SPARSE_RCS_H

#include <stdio.h>

#include "elem.h"
#include "vector.h"
#include "full_r.h"


/*****************************************************************************/

typedef struct {
  int N;
  int m, n;
  elem *v;
  int *j;
  int *r;
} sparse_rcs;

sparse_rcs *sparse_rcs_create(int m, int n, int N);
sparse_rcs *sparse_rcs_create_diag(int m, int n, elem d);
void sparse_rcs_destroy(sparse_rcs **A);
void sparse_rcs_check(const sparse_rcs *A);

sparse_rcs *sparse_rcs_import(const char *filename);
sparse_rcs *sparse_rcs_import3(const char *v_filename, const char *j_filename,
			       const char *r_filename);

void sparse_rcs_export(const char *filename, const sparse_rcs *A);
void sparse_rcs_export3(const char *v_filename, const char *j_filename,
			const char *r_filename, const sparse_rcs *A);

void sparse_rcs_fprintf(FILE *fid, const sparse_rcs *A);
void sparse_rcs_row_fprintf(FILE *fid, const sparse_rcs *A, int row);
void sparse_rcs_rows_fprintf(FILE *fid, const sparse_rcs *A,
			     int row1, int row2);
void sparse_rcs_printf(const sparse_rcs *A);
void sparse_rcs_row_printf(const sparse_rcs *A, int row);
void sparse_rcs_rows_printf(const sparse_rcs *A, int row1, int row2);
void sparse_rcs_debug_printf(const sparse_rcs *A);
void sparse_rcs_debug_fprintf(FILE *fid, const sparse_rcs *A);
void sparse_rcs_r_foreach(const sparse_rcs *A, int i, void (*func)(elem));
void sparse_rcs_mvm(const sparse_rcs *A, const vector *x, vector *y);
void sparse_rcs_mvm_blas(const sparse_rcs *A, const vector *x, vector *y);
void sparse_rcs_mvm_add2col(const sparse_rcs *A, const vector *x,
			    full_r *Y, int j);
void sparse_rcs_mmm(const sparse_rcs *A, const sparse_rcs *B_T, full_r *C);
sparse_rcs *sparse_rcs_transpose(const sparse_rcs *A);
void sparse_rcs_scal(const sparse_rcs *A, elem alpha);


/*****************************************************************************/

typedef struct {
  int N;
  int m, n;
  int *j;
  int *r;
} sparse_binary_rcs;

sparse_binary_rcs *sparse_binary_rcs_import(const char *filename);
void sparse_binary_rcs_export(const char *filename,
                              const sparse_binary_rcs *A);
sparse_binary_rcs *sparse_binary_rcs_create(int m, int n);
void sparse_binary_rcs_destroy(sparse_binary_rcs **A);
void sparse_binary_rcs_printf(const sparse_binary_rcs *A);

#endif
