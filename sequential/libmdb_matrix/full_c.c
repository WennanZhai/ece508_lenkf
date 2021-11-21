#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "full_c.h"
#include "util.h"
#include "eblas.h"


#ifndef NDEBUG
static int check_index(const full_c *A, int i, int j);
#endif

full_c *full_c_create(int m, int n) {
  full_c *A;
  int i;

  A = malloc(sizeof(full_c));
  assert(A);

  A->m = m;
  A->n = n;

  /* Fancy allocation scheme to guarantee the dynamically allocated
   * multidimensional array is contiguous in memory (see C FAQ 6.16) */

  A->v = malloc(sizeof(elem *) * n);
  A->v[0] = malloc(sizeof(elem) * m * n);
  for (i = 1; i < n; i++) {
    A->v[i] = A->v[0] + i * m;
  }

  A->v_vector = A->v[0];

  return A;
}


void full_c_destroy(full_c **A) {
  assert(*A);

  free((*A)->v[0]);
  free((*A)->v);
  free(*A);

  *A = NULL;
}


full_c *full_c_import(const char *filename) {
  FILE *fid;
  int sizeof_elem;
  int m, n;
  int fread_n;
  full_c *A;

  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  fread_n = fread(&sizeof_elem, sizeof(int), 1, fid);
  assert(fread_n == 1);
  assert(sizeof_elem == sizeof(elem));

  fread_n = 0;
  fread_n += fread(&m, sizeof(int), 1, fid);
  fread_n += fread(&n, sizeof(int), 1, fid);
  assert(fread_n == 2);

  assert(m >= 0);
  assert(n >= 0);

  A = full_c_create(m, n);

  fread_n = fread(A->v_vector, sizeof(elem), m*n, fid);
  assert(fread_n == m*n);

  fclose(fid);

  return A;
}


void full_c_export(const char *filename, const full_c *A) {
  full_c_export_partial(filename, A, A->n);
}


void full_c_export_partial(const char *filename, const full_c *A, const int n) {
  FILE *fid;
  size_t fwrite_n;
  int sizeof_elem;

  assert(A);
  assert(filename);
  assert(n >= 0);
  assert(n <= A->n);

  fid = fopen(filename, "w");
  assert(fid);

  sizeof_elem = sizeof(elem);
  fwrite_n = fwrite(&sizeof_elem, sizeof(int), 1, fid);
  assert(fwrite_n == 1);

  fwrite_n = fwrite(&(A->m), sizeof(int), 1, fid);
  fwrite_n += fwrite(&n, sizeof(int), 1, fid);
  assert(fwrite_n == 2);

  fwrite_n = fwrite(A->v_vector, sizeof(elem), A->m * n, fid);
  assert((int) fwrite_n == A->m * n);

  fclose(fid);
}


void full_c_r_foreach(const full_c *A, int i, void (*func)(elem)) {
  int j;

  assert(A);
  assert(func);
  assert(i >= 0 && i < A->m);

  for (j = 0; j < A->n; j++) {
    func(A->v[j][i]);
  }
}

void full_c_foreach(const full_c *A, void (*func)(elem)) {
  int i;

  assert(A);
  assert(func);

  for (i = 0; i < A->m; i++) {
    full_c_r_foreach(A, i, func);
  }
}

void full_c_cols_fprintf(FILE *fid, const full_c *A,
			 const int col1, const int col2) {
  int i, j;

  assert(A);
  assert(col1 >= 0);
  assert(col1 <= col2);
  assert(col2 < A->n);

  for (i = 0; i < A->m; i++) {
    for (j = col1; j <= col2; j++) {
      fprintf_elem_s(fid, full_c_get(A, i, j));
    }
    fprintf(fid, "\n");
  }
}

void full_c_rows_fprintf(FILE *fid, const full_c *A,
			 const int row1, const int row2) {
  int i, j;

  assert(A);
  assert(row1 >= 0);
  assert(row1 <= row2);
  assert(row2 < A->m);

  for (i = row1; i <= row2; i++) {
    for (j = 0; j < A->n; j++) {
      fprintf_elem_s(fid, full_c_get(A, i, j));
    }
    fprintf(fid, "\n");
  }
}

void full_c_submatrix_fprintf(FILE *fid, const full_c *A,
			      const int M, const int N) {
  int i, j;

  assert(A);
  assert(M >= 0);
  assert(M <= A->m);
  assert(N >= 0);
  assert(N <= A->n);

  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      fprintf_elem_s(fid, full_c_get(A, i, j));
    }
    fprintf(fid, "\n");
  }
}

void full_c_fprintf(FILE *fid, const full_c *A) {
  full_c_cols_fprintf(fid, A, 0, A->n-1);
}

void full_c_printf(const full_c *A) {
  full_c_fprintf(stdout, A);
}

void full_c_set(full_c *A, int i, int j, elem v) {
  assert(A);
  assert(check_index(A, i, j));

  A->v[j][i] = v;
}

void full_c_set0(full_c *A) {
  assert(A);

  set0(&(A->v[0][0]), A->m * A->n);
}

elem full_c_get(const full_c *A, int i, int j) {
  assert(A);
  assert(check_index(A, i, j));

  return A->v[j][i];
}

#ifndef NDEBUG
static int check_index(const full_c *A, int i, int j) {
  return (i >= 0 && i < A->m &&
          j >= 0 && j < A->n);
}
#endif

void full_c_export_raw(const full_c *A, const char *filename) {
  FILE *fid;
  size_t n_write;

  assert(A);
  assert(filename);

  fid = fopen(filename, "w");
  assert(fid);

  n_write = fwrite(A->v_vector, sizeof(elem), A->m * A->n, fid);
  assert((int) n_write == A->m * A->n);

  fclose(fid);
}

/* A <- submatrix(B) */
void full_c_get_submatrix(full_c *A, const full_c *B, int i, int j, int m, int n) {
  int k, l;

  assert(A);
  assert(B);
  assert(check_index(B, i, j));
  assert(m > 0 && n > 0);
  assert(i + m - 1 < B->m);
  assert(j + n - 1 < B->n);

  assert(A->m == m);
  assert(A->n == n);

  for (k = j, l = 0; k < j + n; k++, l++) {
    assert(l >= 0 && l < A->n);
    assert(k >= 0 && k < B->n);
    assert(i >= 0 && i < B->m);

    memcpy(&(A->v[l][0]), &(B->v[k][i]), (i + m)*sizeof(elem));
  }
}

/* A <- B */
void full_c_copy(full_c *A, const full_c *B) {
  assert(A);
  assert(B);

  assert((A->m == B->m) && (A->n == B->n));

  memcpy(&(A->v_vector[0]), &(B->v_vector[0]), A->m * A->n * sizeof(elem));
}

/* A <- A * B with B upper triangular */
void full_c_mmult_ut(full_c *A, const full_c *B) {
  assert(A);
  assert(B);
  assert(A->n == B->m);

  etrmm(CblasColMajor, CblasRight, CblasUpper,
        CblasNoTrans, CblasNonUnit,
        A->m, B->n, 1, B->v_vector, B->m, A->v_vector, A->m);
}

/* A <- submatrix(B) with B is CCS (sparse)  */
void full_c_get_submatrix_ccs(full_c *A, const sparse_ccs *B,
                              int i, int j, int m, int n) {
  int k, l, index;

  assert(A);
  assert(B);

  assert(check_index(A, i, j));
  assert(A->m == m);
  assert(A->n == n);

  /* Set all elements of A to 0 */
  memset(&(A->v_vector[0]), 0, A->m * A->n * sizeof(elem));

  for (l = j, k = 0; l < j + n; l++, k++) {
    assert(l >= 0 && l < B->n);
    assert(k >= 0 && k < A->n);

    for (index = B->c[l]; index < B->c[l+1]; index++) {
      if (B->i[index] > i + n - 1) {
        break;
      }
      else if (B->i[index] < i) {
        ;
      }
      else {
        A->v[k][B->i[index]-i] = B->v[index];
      }
    }
  }
}


/* A <- alpha * A */
void full_c_scal(full_c *A, const elem alpha) {
  assert(A);

  escal(A->m * A->n, alpha, A->v_vector, 1);
}


elem full_c_trace(const full_c *A) {
  assert(A);

  return easum(MIN(A->m, A->n), A->v_vector, MIN(A->m, A->n) + 1);
}


int full_c_nnz(const full_c *A) {
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


int full_c_rows_nnz(const full_c *A, const int row1, const int row2) {
  int i, j;
  int nnz;

  assert(A);
  assert(row1 >= 0);
  assert(row2 >= row1);
  assert(row2 <= A->m);

  nnz = 0;

  for (i = row1; i <= row2; i++) {
    for (j = 0; j < A->n; j++) {
      if (A->v[j][i] != 0) {
	nnz++;
      }
    }
  }

  return nnz;
}


/* Matrix matrix multiply */
/* C <- C + A * B */
void full_c_mmm(const full_c *A, const full_c *B, const elem alpha, full_c *C){
  assert(C);
  assert(A);
  assert(B);
  assert(A->m == C->m);
  assert(B->n == C->n);
  assert(A->n == B->m);

  egemm(CblasColMajor, CblasNoTrans, CblasNoTrans, C->m, C->n, A->n, 1,
	A->v_vector, A->m, B->v_vector, B->m, alpha, C->v_vector, C->m);
}


/* Matrix matrix multiply with transpose */
/* C <- C + A * B^T */
void full_c_mmmT(const full_c *A, const full_c *B, const elem alpha, full_c *C){
  assert(C);
  assert(A);
  assert(B);
  assert(A->m == C->m);
  assert(B->m == C->n);
  assert(A->n == B->n);

  egemm(CblasColMajor, CblasNoTrans, CblasTrans, C->m, C->n, A->n, 1,
	A->v_vector, A->m, B->v_vector, B->m, alpha, C->v_vector, C->m);
}
