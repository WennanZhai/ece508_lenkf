#ifndef SPARSE_COO_H
#define SPARSE_COO_H

#include "elem.h"

typedef struct {
  int m, n, N;
  elem *v;
  int *i;
  int *j;
} sparse_coo;


sparse_coo *sparse_coo_create(int m, int n, int N);
void sparse_coo_destroy(sparse_coo **A);
void sparse_coo_printf_raw(const sparse_coo *A);
sparse_coo *sparse_coo_import(char *fname);
void sparse_coo_export(char *fname, const sparse_coo *A);

#endif
