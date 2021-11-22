#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "full_lt_r.h"
#include "elapack.h"


static int lt_r_get_index(const lt_r *A, const int i, const int j);
static int lt_r_get_submatrix_index(const int i, const int j,
				    const int M, const int N);


lt_r *lt_r_create(int m, int n) {
  lt_r *A;

  assert(m > 0);
  assert(n > 0);

  A = malloc(sizeof(lt_r));
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


void lt_r_destroy(lt_r **A) {
  assert(A);
  assert(*A);

  free((*A)->v);
  free(*A);

  *A = NULL;
}


lt_r *lt_r_import(const char *filename) {
  lt_r *A;
  int m, n;
  FILE *fid;
  int r;


  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  r = fread(&m, sizeof(int), 1, fid);
  assert(r == 1);

  r = fread(&n, sizeof(int), 1, fid);
  assert(r == 1);

  A = lt_r_create(m, n);

  r = fread(A->v, sizeof(elem), A->N, fid);
  assert(r == A->N);

  fclose(fid);

  return A;
}


int lt_r_get_index(const lt_r *A, const int i, const int j) {
  if (A->m <= A->n) {
    return i*(i+1)/2 + j;
  }
  else {
    if (i <= A->n) {
      return i*(i+1)/2 + j;
    }
    else {
      return A->n*(A->n+1)/2 + (i-A->n)*A->n + j;
    }
  }
}


elem lt_r_get(const lt_r *A, const int i, const int j) {
  int index;

  assert(A);
  assert(i >= j);

  index = lt_r_get_index(A, i, j);
  return A->v[index];
}


int lt_r_get_submatrix_index(const int i, const int j,
			     const int M, const int N) {
  if (M <= N) {
    return i*(i+1)/2 + j;
  }
  else {
    if (i <= N) {
      return i*(i+1)/2 + j;
    }
    else {
      return N*(N+1)/2 + (i-N)*N + j;
    }
  }
}


elem lt_r_get_submatrix(const lt_r *A, const int i, const int j,
			const int M, const int N) {
  int index;

  assert(A);
  assert(i >= j);

  index = lt_r_get_submatrix_index(i, j, M, N);
  return A->v[index];
}


void lt_r_set(const lt_r *A, const int i, const int j, const elem v) {
  int index;

  assert(A);

  index = lt_r_get_index(A, i, j);
  A->v[index] = v;
}


void lt_r_add(const lt_r *A, const int i, const int j, const elem v) {
  int index;

  assert(A);

  index = lt_r_get_index(A, i, j);
  A->v[index] += v;
}



void lt_r_printf(const lt_r *A) {
  assert(A);

  lt_r_fprintf(stdout, A);
}


void lt_r_fprintf(FILE *fid, const lt_r *A) {
  lt_r_submatrix_fprintf(fid, A, A->m, A->n);
}


void lt_r_submatrix_fprintf(FILE *fid, const lt_r *A,
			    const int M, const int N) {
  int i, j, index;

  assert(fid);
  assert(A);

  index = 0;
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      if (j <= i) {
	fprintf_elem_s(fid, A->v[index]);
	index++;
      }
      else {
	fprintf_elem_s(fid, 0);
      }
    }
    fprintf(fid, "\n");
  }
}


void lt_r_set0(lt_r *A) {
  assert(A);

  set0(A->v, A->N);
}
