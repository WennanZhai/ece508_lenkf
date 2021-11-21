#ifndef SPARSE_CCS_H
#define SPARSE_CCS_H

#include "elem.h"
#include "vector.h"

typedef struct sparse_ccs {
  int N;
  int m, n;
  elem *v;
  int *i;
  int *c;
} sparse_ccs;


sparse_ccs *sparse_ccs_import(const char *filename);
sparse_ccs *sparse_ccs_import3_srt(const char *v_filename,
				   const char *i_filename,
				   const char *c_filename,
				   const int m, const int n);

void sparse_ccs_export(const char *filename, const sparse_ccs *A);
void sparse_ccs_export_mat(const char *filename, const sparse_ccs *A,
			   const char *mat_name);
sparse_ccs *sparse_ccs_create(int m, int n);
void sparse_ccs_destroy(sparse_ccs **A);
void sparse_ccs_printf_c(const sparse_ccs *A, int j);
void sparse_ccs_printf(const sparse_ccs *A);
void sparse_ccs_mvm(const sparse_ccs *A, const vector *x, vector *y);

#endif
