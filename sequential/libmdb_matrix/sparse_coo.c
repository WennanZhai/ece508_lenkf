#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "sparse_coo.h"


sparse_coo *sparse_coo_create(int m, int n, int N) {
  sparse_coo *A;

  assert(m > 0);
  assert(n > 0);
  assert(N > 0);

  A = malloc(sizeof(sparse_coo));
  assert(A);

  A->m = m;
  A->n = n;
  A->N = N;

  A->v = malloc(sizeof(elem) * N);
  assert(A->v);

  A->i = malloc(sizeof(int) * N);
  assert(A->i);

  A->j = malloc(sizeof(int) * N);
  assert(A->j);

  return A;
}


void sparse_coo_destroy(sparse_coo **A) {
  assert(A);
  assert(*A);

  free((*A)->v);
  free((*A)->i);
  free((*A)->j);
  free(*A);

  *A = NULL;
}


void sparse_coo_printf_raw(const sparse_coo *A) {
  int i;

  assert(A);

  for (i = 0; i < A->N; i++) {
    printf("(");
    printf_elem(A->v[i]);
    printf(", %d, %d) ", A->i[i], A->j[i]);
  }
  printf("\n");
}


sparse_coo *sparse_coo_import(char *fname) {
  sparse_coo *A;
  int sizeof_elem;
  int m, n, N;
  FILE *fid;
  int c;

  assert(fname);

  fid = fopen(fname, "r");
  assert(fid);

  c = fread(&sizeof_elem, sizeof(int), 1, fid);
  assert(c == 1);
  assert(sizeof_elem == sizeof(elem));

  c = fread(&m, sizeof(int), 1, fid);
  assert(c == 1);

  c = fread(&n, sizeof(int), 1, fid);
  assert(c == 1);

  c = fread(&N, sizeof(int), 1, fid);
  assert(c == 1);

  A = sparse_coo_create(m, n, N);

  c = fread(A->v, sizeof(elem), N, fid);
  assert(c == N);

  c = fread(A->i, sizeof(int), N, fid);
  assert(c == N);

  c = fread(A->j, sizeof(int), N, fid);
  assert(c == N);

  fclose(fid);

  return A;
}


void sparse_coo_export(char *fname, const sparse_coo *A) {
  FILE *fid;
  int c;
  int sizeof_elem;

  assert(fname);
  assert(A);

  fid = fopen(fname, "w");
  assert(fid);

  sizeof_elem = sizeof(elem);
  c = fwrite(&sizeof_elem, sizeof(int), 1, fid);
  assert(c == 1);

  c = fwrite(&A->m, sizeof(int), 1, fid);
  assert(c == 1);

  c = fwrite(&A->n, sizeof(int), 1, fid);
  assert(c == 1);

  c = fwrite(&A->N, sizeof(int), 1, fid);
  assert(c == 1);

  c = fwrite(A->v, sizeof(elem), A->N, fid);
  assert(c == A->N);

  c = fwrite(A->i, sizeof(int), A->N, fid);
  assert(c == A->N);

  c = fwrite(A->j, sizeof(int), A->N, fid);
  assert(c == A->N);

  fclose(fid);

}
