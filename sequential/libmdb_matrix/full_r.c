#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "full_r.h"
#include "eblas.h"
#include "util.h"


full_r *full_r_create(int m, int n) {
  full_r *A;
  int i;

  A = malloc(sizeof(full_r));
  assert(A);

  A->m = m;
  A->n = n;

  /* Fancy allocation scheme to guarantee the dynamically allocated
   * multidimensional array is contiguous in memory (see C FAQ 6.16) */

  A->v = malloc(sizeof(elem *) * m);
  A->v[0] = malloc(sizeof(elem) * m * n);
  for (i = 1; i < m; i++) {
    A->v[i] = A->v[0] + i * n;
  }

  A->v_vector = A->v[0];

  return A;
}

void full_r_destroy(full_r **A) {
  assert(A);
  assert(*A);
  assert((*A)->v);
  assert((*A)->v[0]);

  free((*A)->v[0]);
  free((*A)->v);
  free(*A);

  *A = NULL;
}


void full_r_r_foreach(const full_r *A, int i, void (*func)(elem)) {
  int j;

  assert(A);
  assert(func);
  assert(i >= 0 && i < A->m);

  for (j = 0; j < A->n; j++) {
    func(A->v[i][j]);
  }
}

void full_r_foreach(const full_r *A, void (*func)(elem)) {
  int i;

  assert(A);
  assert(func);

  for (i = 0; i < A->m; i++) {
    full_r_r_foreach(A, i, func);
  }
}

void full_r_fprintf(FILE *fid, const full_r *A) {
  full_r_submatrix_fprintf(fid, A, A->m, A->n);
}


void full_r_rows_fprintf(FILE *fid, const full_r *A,
			 const int i1, const int i2) {
  int i, j;

  assert(fid);
  assert(A);
  assert(i1 >= 0);
  assert(i1 <= i2);
  assert(i2 <= A->m - 1);

  for (i = i1; i <= i2; i++) {
    for (j = 0; j < A->n; j++) {
      fprintf_elem_s(fid, A->v[i][j]);
    }
    fprintf(fid, "\n");
  }
}


void full_r_submatrix_fprintf(FILE *fid, const full_r *A,
			      const int M, const int N) {
  int i;
  int j;

  assert(fid);
  assert(A);
  assert(M <= A->m);
  assert(N <= A->n);

  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      fprintf_elem_s(fid, A->v[i][j]);
    }
    fprintf(fid, "\n");
  }
}


void full_r_printf(const full_r *A) {
  full_r_fprintf(stdout, A);
}

void full_r_set(full_r *A, int i, int j, elem v) {
  assert(A);
  assert(full_r_check_index(A, i, j));

  A->v[i][j] = v;
}

elem full_r_get(const full_r *A, int i, int j) {
  assert(A);
  assert(full_r_check_index(A, i, j));

  return A->v[i][j];
}

int full_r_check_index(const full_r *A, int i, int j) {
  return ((i >= 0) && (i < A->m) &&
          (j >= 0) && (j < A->n));
}

full_r *full_r_import(const char *filename) {
  full_r *A;
  int sizeof_elem;
  int m, n;
  int fread_n;
  FILE *fid;

  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  fread_n = fread(&sizeof_elem, sizeof(int), 1, fid);
  assert(fread_n == 1);
  assert(sizeof_elem = sizeof(elem));

  fread_n = 0;
  fread_n += fread(&m, sizeof(int), 1, fid);
  fread_n += fread(&n, sizeof(int), 1, fid);
  assert(fread_n == 2);

  assert(m >= 0);
  assert(n >= 0);

  A = full_r_create(m, n);

  fread_n = fread(A->v_vector, sizeof(elem), m*n, fid);
  assert(fread_n == m*n);

  fclose(fid);

  return A;
}

void full_r_export(const char *filename, const full_r *A) {
  int i, j;
  int sizeof_elem;
  FILE *fid;
  size_t fwrite_n;

  assert(A);
  assert(filename);

  fid = fopen(filename, "w");
  assert(fid);

  sizeof_elem = sizeof(elem);
  fwrite_n = fwrite(&sizeof_elem, sizeof(int), 1, fid);
  assert(fwrite_n == 1);

  fwrite_n = fwrite(&(A->m), sizeof(int), 1, fid);
  fwrite_n += fwrite(&(A->n), sizeof(int), 1, fid);
  assert(fwrite_n == 2);

  for (i = 0; i < A->m; i++) {
    for (j = 0; j < A->n; j++) {
      elem *v;
      v = &(A->v[i][j]);

      fwrite_n = fwrite(v, sizeof(elem), 1, fid);
      assert(fwrite_n == 1);
    }
  }

  fclose(fid);
}


elem *full_r_get_ptr(const full_r *A, int i, int j) {
  assert(A);
  assert(full_r_check_index(A, i, j));

  return &(A->v[i][j]);
}

void full_r_set0(full_r *A) {
  assert(A);

  set0(&(A->v[0][0]), A->m * A->n);
}

void full_r_setI(full_r *A) {
  int i;

  assert(A);

  full_r_set0(A);

  for (i = 0; i < MIN(A->m, A->n); i++) {
    full_r_set(A, i, i, 1);
  }
}

/* Y <- X */
void full_r_copy(full_r *Y, const full_r *X) {
  assert(Y);
  assert(X);

  assert(Y->m == X->m);
  assert(Y->n == X->n);

  ecopy(X->m * X->n, X->v[0], 1, Y->v[0], 1);
}


elem full_r_trace(const full_r *A) {
  assert(A);
  assert(A->m == A->n);

  return easum(A->m, A->v_vector, A->m + 1);
}


void full_r_scal(full_r *A, const elem alpha) {
  assert(A);

  escal(A->m * A->n, alpha, A->v_vector, 1);
}


int full_r_nnz(const full_r *A) {
  int i;
  int nnz;

  assert(A);

  nnz = 0;
  for (i = 0; i < A->m * A->n; i++) {
    if (A->v_vector[i] != 0) {
      nnz++;
    }
  }

  return nnz;
}


/* Matrix matrix multiply */
/* C <- C + A * B */
void full_r_mmm(const full_r *A, const full_r *B, const elem alpha, full_r *C){
  assert(C);
  assert(A);
  assert(B);
  assert(A->m == C->m);
  assert(B->n == C->n);
  assert(A->n == B->m);

  egemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C->m, C->n, A->n, 1,
	A->v_vector, A->n, B->v_vector, B->n, alpha, C->v_vector, C->n);
}


/* Matrix matrix multiply with transpose */
/* C <- C + A * B^T */
void full_r_mmmT(const full_r *A, const full_r *B, const elem alpha, full_r *C){
  assert(C);
  assert(A);
  assert(B);
  assert(A->m == C->m);
  assert(B->m == C->n);
  assert(A->n == B->n);

  egemm(CblasRowMajor, CblasNoTrans, CblasTrans, C->m, C->n, A->n, 1,
	A->v_vector, A->n, B->v_vector, B->n, alpha, C->v_vector, C->n);
}
