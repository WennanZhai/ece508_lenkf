#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "sparse_ccs.h"
#include "util.h"


sparse_ccs *sparse_ccs_create(int m, int n) {
  sparse_ccs *A;

  A = malloc(sizeof(sparse_ccs));
  assert(A);

  A->m = m;
  A->n = n;
  A->N = 0;
  A->v = NULL;
  A->i = NULL;
  A->c = NULL;
  
  return A;
}

void sparse_ccs_destroy(sparse_ccs **A) {
  assert(*A);

  free((*A)->v);
  free((*A)->i);
  free((*A)->c);
  free(*A);
  *A = NULL;
}

sparse_ccs *sparse_ccs_import(const char *filename) {
  FILE *fid;
  sparse_ccs *A;
  int count;
  int sizeof_elem;
  
  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  A = malloc(sizeof(sparse_ccs));
  assert(A);

  count = fread(&sizeof_elem, sizeof(int), 1, fid);
  assert(count == 1);
  assert(sizeof_elem == sizeof(elem));
  
  count = fread(&(A->m), sizeof(int), 1, fid);
  count += fread(&(A->n), sizeof(int), 1, fid);
  count += fread(&(A->N), sizeof(int), 1, fid);
  assert(count == 3);

  A->v = malloc(sizeof(elem) * A->N);
  A->i = malloc(sizeof(int) * A->N);
  A->c = malloc(sizeof(int) * (A->n + 1));

  assert(A->v && A->i && A->c);
  
  count = fread(A->v, sizeof(elem), A->N, fid);
  count += fread(A->i, sizeof(int), A->N, fid);
  count += fread(A->c, sizeof(int), A->n + 1, fid);

  assert(count == A->N + A->N + A->n + 1);

  fclose(fid);
  
  return A;
}

sparse_ccs *sparse_ccs_import3_srt(const char *v_filename,
				   const char *i_filename,
				   const char *c_filename,
				   const int m, const int n) {
  FILE *fid;
  sparse_ccs *A;
  int count;
  
  assert(v_filename);
  assert(i_filename);
  assert(c_filename);
  assert(m > 0);
  assert(n > 0);

  A = malloc(sizeof(sparse_ccs));
  assert(A);

  A->m = m;
  A->n = n;

  A->c = malloc(sizeof(int) * (n + 1));
  assert(A->c);

  fid = fopen(c_filename, "r");
  assert(fid);
  
  count = fread(A->c, sizeof(int), n + 1, fid);
  assert(count == n + 1);

  fclose(fid);


  A->N = A->c[n];
  assert(A->N > 0);

  A->v = malloc(sizeof(elem) * A->N);
  assert(A->v);

  A->i = malloc(sizeof(int) * A->N);
  assert(A->i);

  
  fid = fopen(v_filename, "r");
  assert(fid);

  count = fread(A->v, sizeof(elem), A->N, fid);
  assert(count == A->N);
  
  fclose(fid);

  
  fid = fopen(i_filename, "r");
  assert(fid);

  count = fread(A->i, sizeof(int), A->N, fid);
  assert(count == A->N);
  
  fclose(fid);
  
  return A;
}


void sparse_ccs_export(const char *filename, const sparse_ccs *A) {
  FILE *fid;
  int count;
  int sizeof_elem;
  
  assert(filename);
  assert(A);
  
  fid = fopen(filename, "w");
  assert(fid);

  sizeof_elem = sizeof(elem);
  count = fwrite(&sizeof_elem, sizeof(int), 1, fid);
  assert(count == 1);
  
  count = fwrite(&(A->m), sizeof(int), 1, fid);
  count += fwrite(&(A->n), sizeof(int), 1, fid);
  count += fwrite(&(A->N), sizeof(int), 1, fid);
  assert(count == 3);
  
  count = fwrite(A->v, sizeof(elem), A->N, fid);
  count += fwrite(A->i, sizeof(int), A->N, fid);
  count += fwrite(A->c, sizeof(int), A->n + 1, fid);

  assert(count == A->N + A->N + A->n + 1);
}


void sparse_ccs_printf_c(const sparse_ccs *A, int j) {
  int i, num_elem, index;
  
  assert(A);
  assert(j >= 0 && j < A->n);

  num_elem = A->c[j+1] - A->c[j];
  assert(num_elem >= 0);

  index = A->c[j];
  
  for (i = 0; i < A->m; i++) {
    if (i == A->i[index]) {
      printf_elem_n(A->v[index]);
      index++;
    }
    else {
      printf("%+f\n", 0.0);
    }
  }
}

void sparse_ccs_printf(const sparse_ccs *A) {
  int j;

  assert(A);

  for (j = 0; j < A->n; j++) {
    printf("column %d\n", j);
    sparse_ccs_printf_c(A, j);

    if (j != A->n - 1) {
      printf("\n");
    }
  }
}


void sparse_ccs_mvm(const sparse_ccs *A, const vector *x, vector *y) {
  int j, index;
  
  assert(A);
  assert(x);
  assert(y);
  assert(A->m == y->n);
  assert(A->n == x->n);

  vector_set0(y);
  
  for (j = 0; j < A->n; j++) {
    for (index = A->c[j]; index < A->c[j+1]; index++) {
      y->v[A->i[index]] += A->v[index] * x->v[j];
    }
  }
}
