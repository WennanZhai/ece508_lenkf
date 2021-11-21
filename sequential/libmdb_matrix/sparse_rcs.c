#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "sparse_rcs.h"
#include "eblas.h"
#include "util.h"


/*****************************************************************************/

sparse_rcs *sparse_rcs_create(int m, int n, int N) {
  sparse_rcs *A;

  assert(m >= 0);
  assert(n >= 0);
  assert(N >= 0);

  A = malloc(sizeof(sparse_rcs));
  assert(A);

  A->m = m;
  A->n = n;
  A->N = N;

  if (m > 0) {
    A->r = malloc(sizeof(int) * (m+1));
    assert(A->r);
  }
  else {
    A->r = NULL;
  }

  if (N == 0) {
    A->v = NULL;
    A->j = NULL;
  }
  else {
    A->v = malloc(sizeof(elem) * N);
    assert(A->v);

    A->j = malloc(sizeof(elem) * N);
    assert(A->j);
  }

  return A;
}


sparse_rcs *sparse_rcs_create_diag(int m, int n, elem d) {
  sparse_rcs *A;
  int i;

  assert(m > 0);
  assert(n > 0);

  A = sparse_rcs_create(m, n, MIN(m, n));

  for (i = 0; i < A->m; i++) {
    A->v[i] = d;
    A->j[i] = i;
    A->r[i] = i;
  }
  A->r[m] = A->N;

  return A;
}


void sparse_rcs_destroy(sparse_rcs **A) {
  assert(*A);

  /* No memory is allocated for v or j if A is an all zero matrix */
  if ((*A)->N > 0) {
    free((*A)->v);
    free((*A)->j);
  }
  else {
    assert((*A)->v == NULL);
    assert((*A)->j == NULL);
  }

  /* No memory is allocated for r if A has 0 rows */
  if ((*A)->m > 0) {
    free((*A)->r);
  }
  else {
    assert((*A)->r == NULL);
  }

  free(*A);

  *A = NULL;
}

void sparse_rcs_check(const sparse_rcs *A) {
  int i;

  assert(A);
  assert(A->m >= 0);
  assert(A->n >= 0);
  assert(A->N >= 0);

  if (A->m == 0) {
    assert(A->N == 0);
  }

  if (A->m > 0) {
    assert(A->N == A->r[A->m]);
  }

  for (i = 0; i < A->N; i++) {
    assert(A->j[i] >= 0 && A->j[i] < A->n);
  }

  for (i = 0; i < A->m; i++) {
    assert(A->r[i] <= A->r[i+1]);
  }
}

sparse_rcs *sparse_rcs_import(const char *filename) {
  FILE *fid;
  sparse_rcs *A = NULL;
  int sizeof_elem;
  int m, n, N;
  int r;

  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  r = fread(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);
  assert(sizeof_elem == sizeof(elem));

  r = fread(&m, sizeof(int), 1, fid);
  assert(r == 1);

  r = fread(&n, sizeof(int), 1, fid);
  assert(r == 1);

  r = fread(&N, sizeof(int), 1, fid);
  assert(r == 1);

  A = sparse_rcs_create(m, n, N);

  if (m > 0) {
    r = fread(A->v, sizeof(elem), A->N, fid);
    assert(r == A->N);

    r = fread(A->j, sizeof(int), A->N, fid);
    assert(r == A->N);

    r = fread(A->r, sizeof(int), A->m + 1, fid);
    assert(r == A->m + 1);
  }
  else {
    assert(N == 0);
  }

  fclose(fid);

#ifndef NDEBUG
  sparse_rcs_check(A);
#endif

  return A;
}

sparse_rcs *sparse_rcs_import3(const char *v_filename, const char *j_filename,
			       const char *r_filename) {
  sparse_rcs *A;
  int m, n, N;
  FILE *fid;
  int r;

  assert(v_filename);
  assert(j_filename);
  assert(r_filename);

  fid = fopen(r_filename, "r");
  assert(fid);

  r = fseek(fid, -3*sizeof(int), SEEK_END);
  assert(r == 0);

  r = fread(&N, sizeof(int), 1, fid);
  assert(r == 1);

  r = fread(&m, sizeof(int), 1, fid);
  assert(r == 1);

  r = fread(&n, sizeof(int), 1, fid);
  assert(r == 1);

  A = sparse_rcs_create(m, n, N);

  rewind(fid);
  r = fread(A->r, sizeof(int), m+1, fid);
  assert(r == m+1);
  fclose(fid);

  fid = fopen(v_filename, "r");
  assert(fid);

  r = fread(A->v, sizeof(elem), N, fid);
  assert(r == N);
  fclose(fid);

  fid = fopen(j_filename, "r");
  assert(fid);

  r = fread(A->j, sizeof(int), N, fid);
  assert(r == N);
  fclose(fid);

#ifdef NDEBUG
  sparse_rcs_check(A);
#endif

  return A;
}

void sparse_rcs_export(const char *filename, const sparse_rcs *A) {
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
  count += fwrite(A->j, sizeof(int), A->N, fid);
  count += fwrite(A->r, sizeof(int), A->m + 1, fid);

  assert(count == A->N + A->N + A->m + 1);

  fclose(fid);
}

void sparse_rcs_export3(const char *v_filename, const char *j_filename,
			const char *r_filename, const sparse_rcs *A) {
  FILE *fid;
  int r;

  assert(v_filename);
  assert(j_filename);
  assert(r_filename);
  assert(A);

  fid = fopen(r_filename, "w");
  assert(fid);

  r = fwrite(A->r, sizeof(int), A->m + 1, fid);
  assert(r == A->m + 1);

  r = fwrite(&A->m, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&A->n, sizeof(int), 1, fid);
  assert(r == 1);
  fclose(fid);

  fid = fopen(v_filename, "w");
  assert(fid);

  r = fwrite(A->v, sizeof(elem), A->N, fid);
  assert(r == A->N);
  fclose(fid);

  fid = fopen(j_filename, "w");
  assert(fid);

  r = fwrite(A->j, sizeof(int), A->N, fid);
  assert(r == A->N);
  fclose(fid);
}


void sparse_rcs_fprintf(FILE *fid, const sparse_rcs *A) {
  int i, j, A_index, n_elem;


  for (i = 0; i < A->m; i++) {
    n_elem = A->r[i+1] - A->r[i];

    if (n_elem == 0) {
      for (j = 0; j < A->n; j++) {
        fprintf_elem_s(fid, (elem) 0);
      }
    }
    else {
      A_index = A->r[i];
      for (j = 0; j < A->n; j++) {
	assert(A_index <= A->r[i+1]);
	if ((A_index == A->r[i+1]) || (j != A->j[A_index])) {
	  fprintf_elem_s(fid, (elem) 0);
	}
	else {
	  fprintf_elem_s(fid, A->v[A_index]);
	  A_index++;
	}
      }
    }

    fprintf(fid, "\n");
  }
}

void sparse_rcs_row_fprintf(FILE *fid, const sparse_rcs *A, int row) {
  int n_elem, j, row_index, row_count;

  assert(A);
  assert(row >= 0 && row < A->m);

  n_elem = A->r[row+1] - A->r[row];

  if (n_elem == 0) {
    for (j = 0; j < A->n; j++) {
      fprintf_elem_s(fid, (elem) 0);
    }
  }
  else {
    row_index = A->r[row];
    row_count = 0;
    for (j = 0; j < A->n; j++) {
      if (row_count < n_elem && A->j[row_index] == j) {
	fprintf_elem_s(fid, A->v[row_index]);
	row_index++;
	row_count++;
      }
      else {
	fprintf_elem_s(fid, (elem) 0);
      }
    }
  }

  fprintf(fid, "\n");
}

void sparse_rcs_rows_fprintf(FILE *fid, const sparse_rcs *A,
			     int row1, int row2)  {
  int i;

  assert(fid);
  assert(A);
  assert(row1 >= 0 && row1 <= row2);
  assert(row2 < A->m);

  for (i = row1; i <= row2; i++) {
    sparse_rcs_row_fprintf(fid, A, i);
  }
}

void sparse_rcs_printf(const sparse_rcs *A) {
  sparse_rcs_fprintf(stdout, A);
}

void sparse_rcs_row_printf(const sparse_rcs *A, int row) {
  sparse_rcs_row_fprintf(stdout, A, row);
}

void sparse_rcs_rows_printf(const sparse_rcs *A, int row1, int row2) {
  sparse_rcs_rows_fprintf(stdout, A, row1, row2);
}

void sparse_rcs_r_foreach(const sparse_rcs *A, int i, void (*func)(elem)) {
  int start, stop, k;

  assert(A);
  assert(func);
  assert(i >= 0 && i < A->m);

  start = A->r[i];
  stop = A->r[i+1];

  for (k = start; i < stop; i++) {
    func(A->v[k]);
  }
}

void sparse_rcs_debug_printf(const sparse_rcs *A) {
  sparse_rcs_debug_fprintf(stdout, A);
}

void sparse_rcs_debug_fprintf(FILE *fid, const sparse_rcs *A) {
  int i;

  assert(fid);
  assert(A);

  fprintf(fid, "N=%d\n", A->N);
  fprintf(fid, "m=%d\tn=%d\n", A->m, A->n);

  fprintf(fid, "v:\n");
  for (i = 0; i < A->N; i++) {
    fprintf(fid, "%f ", A->v[i]);
  }
  fprintf(fid, "\n\n");

  fprintf(fid, "j:\n");
  for (i = 0; i < A->N; i++) {
    fprintf(fid, "%d ", A->j[i]);
  }
  fprintf(fid, "\n\n");

  fprintf(fid, "r:\n");
  for (i = 0; i < A->m + 1; i++) {
    fprintf(fid, "%d ", A->r[i]);
  }
  fprintf(fid, "\n");
}

/* Sparse matrix vector multiply */
/* y <- A x */
void sparse_rcs_mvm(const sparse_rcs *A, const vector *x, vector *y) {
  int i, l;
  int n_elem;

  assert(A);
  assert(x);
  assert(y);

  assert(A->n == x->n);
  assert(A->m == y->n);

  vector_set0(y);

  for (i = 0; i < A->m; i++) {
    n_elem = A->r[i+1] - A->r[i];

    for (l = 0; l < n_elem; l++) {
      y->v[i] += A->v[A->r[i] + l] * x->v[A->j[A->r[i] + l]];
    }
  }
}

/* Sparse matrix vector multiply */
/* Y(:,j) <- Y(:,j) + A x */
void sparse_rcs_mvm_add2col(const sparse_rcs *A, const vector *x,
			    full_r *Y, int j) {
  int i, n_elem, l;

  assert(A);
  assert(x);
  assert(Y);

  assert(A->n == x->n);
  assert(A->m == Y->m);
  assert(j >= 0 && j < Y->n);

  for (i = 0; i < A->m; i++) {
    n_elem = A->r[i+1] - A->r[i];

    for (l = 0; l < n_elem; l++) {
      Y->v[i][j] += A->v[A->r[i] + l] * x->v[A->j[A->r[i] + l]];
    }
  }
}

/* Sparse matrix vector multiply making use of BLAS */
/* y <- A x */
void sparse_rcs_mvm_blas(const sparse_rcs *A, const vector *x, vector *y) {
  int i, l;
  int n_elem;

  vector *row;
  vector *x_portion;

  assert(A);
  assert(x);
  assert(y);

  assert(A->n == x->n);
  assert(A->m == y->n);

  row = vector_create(A->n);
  x_portion = vector_create(x->n);

  for (i = 0; i < A->m; i++) {
    n_elem = A->r[i+1] - A->r[i];

    for (l = 0; l < n_elem; l++) {
      row->v[l] = A->v[A->r[i] + l];
      x_portion->v[l] = x->v[A->j[A->r[i] + l]];
    }

    edot(n_elem, row->v, 1, x_portion->v, 1);
  }

  vector_destroy(&row);
  vector_destroy(&x_portion);
}


/* Compute the product C <= A * B */
/* Note that the transpose of B is passed to this function */
void sparse_rcs_mmm(const sparse_rcs *A, const sparse_rcs *B_T, full_r *C) {
  int i, j;
  int index_A, index_B_T;

  assert(A);
  assert(B_T);
  assert(C);
  assert(A->m == C->m);
  assert(B_T->m == C->n);
  assert(A->n == B_T->n);

  full_r_set0(C);

  for (i = 0; i < C->m; i++) {
    for (j = 0; j < C->n; j++) {
      index_A = A->r[i];

      for (index_B_T = B_T->r[j]; index_B_T < B_T->r[j+1]; index_B_T++) {
	while (A->j[index_A] < B_T->j[index_B_T]) {
	  index_A++;

	  if (index_A >= A->r[i+1]) {
	    break;
	  }
	}

	if (index_A >= A->r[i+1]) {
	  break;
	}

	if (A->j[index_A] == B_T->j[index_B_T]) {
	  C->v[i][j] += A->v[index_A] * B_T->v[index_B_T];
	}
      }
    }
  }
}


/* out <- A^T */
sparse_rcs *sparse_rcs_transpose(const sparse_rcs *A) {
  sparse_rcs *B;
  ivector *col_count;
  int i, j, index;
  typedef struct {
    elem v;
    int i;
  } elem_int_pair;
  elem_int_pair **elem_int_pair_list;

  assert(A);


  col_count = ivector_create(A->n);
  ivector_set0(col_count);

  for (i = 0; i < A->N; i++) {
    col_count->v[A->j[i]]++;
  }

  elem_int_pair_list = malloc(A->n * sizeof(elem_int_pair *));
  assert(elem_int_pair_list);

  for (i = 0; i < A->n; i++) {
    elem_int_pair_list[i] = malloc(col_count->v[i] * sizeof(elem_int_pair));
    assert(elem_int_pair_list[i]);
  }

  ivector_set0(col_count);

  for (i = 0; i < A->m; i++) {
    for (j = A->r[i]; j < A->r[i+1]; j++) {
      elem_int_pair_list[A->j[j]][col_count->v[A->j[j]]].v = A->v[j];
      elem_int_pair_list[A->j[j]][col_count->v[A->j[j]]].i = i;
      col_count->v[A->j[j]]++;
    }
  }

  B = sparse_rcs_create(A->n, A->m, A->N);

  index = 0;
  for (i = 0; i < B->m; i++) {
    B->r[i] = index;
    for (j = 0; j < col_count->v[i]; j++) {
      B->v[index] = elem_int_pair_list[i][j].v;
      B->j[index] = elem_int_pair_list[i][j].i;
      index++;
    }
  }
  B->r[B->m] = index;


  ivector_destroy(&col_count);

  for (i = 0; i < A->n; i++) {
    free(elem_int_pair_list[i]);
  }
  free(elem_int_pair_list);

  return B;
}


/* A <- alpha * A */
void sparse_rcs_scal(const sparse_rcs *A, elem alpha) {
  assert(A);

  escal(A->N, alpha, A->v, 1);
}


/*****************************************************************************/

sparse_binary_rcs *sparse_binary_rcs_import(const char *filename) {
  FILE *fid;
  sparse_binary_rcs *A = NULL;
  int count;

  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  A = malloc(sizeof(sparse_binary_rcs));
  assert(A);

  count = fread(&(A->m), sizeof(int), 1, fid);
  count += fread(&(A->n), sizeof(int), 1, fid);
  count += fread(&(A->N), sizeof(int), 1, fid);
  assert(count == 3);

  A->j = malloc(sizeof(int) * A->N);
  A->r = malloc(sizeof(int) * (A->m + 1));

  assert(A->j && A->r);

  count = fread(A->j, sizeof(int), A->N, fid);
  count += fread(A->r, sizeof(int), A->m + 1, fid);

  assert(count == A->N + A->m + 1);

  fclose(fid);

  return A;
}

void sparse_binary_rcs_export(const char *filename,
                              const sparse_binary_rcs *A) {
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

  count = fwrite(A->j, sizeof(int), A->N, fid);
  count += fwrite(A->r, sizeof(int), A->m + 1, fid);

  assert(count == A->N + A->m + 1);
}

sparse_binary_rcs *sparse_binary_rcs_create(int m, int n) {
  sparse_binary_rcs *A;

  A = malloc(sizeof(sparse_binary_rcs));
  assert(A);

  A->m = m;
  A->n = n;
  A->N = 0;
  A->j = NULL;
  A->r = NULL;

  return A;
}

void sparse_binary_rcs_destroy(sparse_binary_rcs **A) {
  assert(*A);

  free((*A)->j);
  free((*A)->r);
  free(*A);

  *A = NULL;
}

void sparse_binary_rcs_printf(const sparse_binary_rcs *A) {
  int i, j, A_index;

  A_index = 0;

  for (i = 0; i < A->m; i++) {
    int n_r = A->r[i+1] - A->r[i];

    if (n_r == 0) {
      for (j = 0; j < A->n; j++) {
        printf_elem_s((elem) 0);
      }
    }
    else {
      for (j = 0; j < A->n; j++) {
        if (A->j[A_index] == j) {
          printf_elem_s((elem) 1);
          A_index++;
        }
        else {
          printf_elem_s((elem) 0);
        }
      }
    }

    printf("\n");
  }
}
