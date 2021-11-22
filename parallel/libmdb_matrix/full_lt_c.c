#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "util.h"
#include "full_lt_c.h"
#include "elapack.h"


static int lt_c_get_index(const lt_c *A, const int i, const int j);
static int lt_c_get_submatrix_index(const int i, const int j, const int M);


lt_c *lt_c_create(int m, int n) {
  lt_c *A;

  assert(m > 0 && n > 0);

  A = malloc(sizeof(lt_c));
  assert(A);

  A->m = m;
  A->n = n;

  if (m <= n) {
    A->N = m*(m+1)/2;
  }
  else {
    A->N = n*(n+1)/2 + (m - n)*n;
  }

  A->v = malloc(sizeof(elem) * A->N);
  assert(A->v);

  return A;
}

void lt_c_destroy(lt_c **A) {
  assert(*A);

  free((*A)->v);
  free(*A);
  *A = NULL;
}


lt_c *lt_c_import(const char *filename) {
  lt_c *A;
  int m, n;
  int r;

  FILE *fid;

  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  r = fread(&m, sizeof(int), 1, fid);
  assert(r == 1);

  r = fread(&n, sizeof(int), 1, fid);
  assert(r == 1);

  A = lt_c_create(m, n);

  r = fread(A->v, sizeof(elem), A->N, fid);
  assert(r == A->N);

  fclose(fid);

  return A;
}


int lt_c_get_index(const lt_c *A, const int i, const int j) {
  return j*(2*A->m-j-1)/2 + i;
}


elem lt_c_get(const lt_c *A, const int i, const int j) {
  assert(A);
  assert(i >= 0 && i < A->m);
  assert(j >= 0 && j < A->n);
  assert(i >= j);

  return A->v[lt_c_get_index(A, i, j)];
}

static int lt_c_get_submatrix_index(const int i, const int j, const int M) {
  return j*(2*M-j-1)/2 + i;
}

elem lt_c_get_submatrix(const lt_c *A, int i, int j, const int M) {
  assert(A);
  assert(i >= 0 && i < M);
  assert(j >= 0 && j < M);
  assert(i >= j);

  return A->v[lt_c_get_submatrix_index(i, j, M)];
}


void lt_c_set(const lt_c *A, int i, int j, elem v) {
  assert(A);
  assert(i >= 0 && i < A->m);
  assert(j >= 0 && j < A->n);
  assert(i >= j);

  A->v[lt_c_get_index(A, i, j)] = v;
}


void lt_c_add(const lt_c *A, int i, int j, elem v) {
  assert(A);
  assert(i >= 0 && i < A->m);
  assert(j >= 0 && j < A->n);
  assert(i >= j);

  A->v[lt_c_get_index(A, i, j)] += v;
}


void lt_c_fprintf(FILE *fid, const lt_c *A) {
  int i, j;

  assert(fid);
  assert(A);

  for (i = 0; i < A->m; i++) {
    for (j = 0; j < A->n; j++) {
      if (j <= i) {
	fprintf_elem_s(fid, lt_c_get(A, i, j));
      }
      else {
	fprintf_elem_s(fid, 0);
      }
    }
    fprintf(fid, "\n");
  }
}


void lt_c_submatrix_fprintf(FILE *fid, const lt_c *A, const int M) {
  int i, j;

  assert(fid);
  assert(A);
  assert(M >= 0 && M <= MIN(A->m, A->n));

  for (i = 0; i < M; i++) {
    for (j = 0; j < M; j++) {
      if (j <= i) {
	fprintf_elem_s(fid, lt_c_get_submatrix(A, i, j, M));
      }
      else {
	fprintf_elem_s(fid, 0);
      }
    }
    fprintf(fid, "\n");
  }
}


void lt_c_printf(const lt_c *A) {
  lt_c_fprintf(stdout, A);
}


void lt_c_set0(lt_c *A) {
  assert(A);

  set0(A->v, A->N);
}


void lt_c_eppsv(lt_c *A, full_c *B) {
  assert(A);
  assert(B);
  assert(A->m == A->n);
  assert(A->n == B->m);

  eppsv(LAPACK_COL_MAJOR, 'L', A->n, B->n, A->v, B->v_vector, B->m);
}
