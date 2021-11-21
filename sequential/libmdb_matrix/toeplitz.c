#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include "toeplitz.h"
#include "eblas.h"


/*************
 * s_toeplitz
 *************/


s_toeplitz *s_toeplitz_create(int n, int nnz_row0) {
  s_toeplitz *A;

  assert(n > 0);
  assert(nnz_row0 > 0);

  A = malloc(sizeof(s_toeplitz));
  assert(A);

  A->n = n;
  A->nnz_row0 = nnz_row0;

  A->v_row0 = malloc(sizeof(elem) * nnz_row0);
  assert(A->v_row0);

  A->j_row0 = malloc(sizeof(elem) * nnz_row0);
  assert(A->j_row0);

  /* A row of the symmetric Toeplitz can have at most 2*nnz_row - 1
     elements */
  A->v = malloc(sizeof(elem) * (2*nnz_row0 - 1));
  assert(A->v);

  A->j = malloc(sizeof(int) * (2*nnz_row0 - 1));
  assert(A->j);

  A->row = -1;
  A->nnz = 0;

  return A;
}


void s_toeplitz_destroy(s_toeplitz **A) {
  assert(A);
  assert(*A);

  free((*A)->v_row0);
  free((*A)->j_row0);
  free((*A)->v);
  free((*A)->j);
  free(*A);

  *A = NULL;
}


void s_toeplitz_get_row(s_toeplitz *A, int row) {
  int i, index;

  assert(A);
  assert(row >= 0 && row < A->n);

  if (A->row == row) {
    return;
  }

  A->row = row;

  index = 0;
  /* Build portion of the row to the left of the diagonal */
  for (i = A->nnz_row0 - 1; i >= 1; i--) {
    if (-A->j_row0[i] + row >= 0) {
      A->v[index] = A->v_row0[i];
      A->j[index] = -A->j_row0[i] + row;
      index++;
    }
  }

  /* Build portion of the row to the right of the diagonal
     (including the diagonal) */
  for (i = 0; i < A->nnz_row0; i++) {
    if (A->j_row0[i] + row < A->n) {
      A->v[index] = A->v_row0[i];
      A->j[index] = A->j_row0[i] + row;
      index++;
    }
    else {
      break;
    }
  }

  A->nnz = index;
}


void s_toeplitz_printf_row(s_toeplitz *A, int row) {
  assert(A);

  s_toeplitz_get_row(A, row);

  s_toeplitz_printf_current_row(A);
}


void s_toeplitz_printf_current_row(s_toeplitz *A) {
  int i, index;

  assert(A);

  index = 0;
  for (i = 0; i < A->n; i++) {
    if (index < A->nnz && i == A->j[index]) {
      printf_elem_s(A->v[index]);
      index++;
    }
    else {
      printf_elem_s((elem) 0);
    }
  }
}


void s_toeplitz_printf(s_toeplitz *A) {
  int row;

  assert(A);

  for (row = 0; row < A->n; row++) {
    s_toeplitz_printf_row(A, row);

    if (row != A->n - 1) {
      printf("\n");
    }
  }
}


s_toeplitz *s_toeplitz_import(const char *filename) {
  s_toeplitz *A;
  FILE *fid;

  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  A = s_toeplitz_import_fid(fid);

  fclose(fid);

  return A;
}


s_toeplitz *s_toeplitz_import_fid(FILE *fid) {
  s_toeplitz *A;
  int N, nnz_row0;
  int r;
  int sizeof_elem;

  assert(fid);

  r = fread(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);
  assert(sizeof_elem == sizeof(elem));

  r = fread(&N, sizeof(int), 1, fid);
  assert(r == 1);

  r = fread(&nnz_row0, sizeof(int), 1, fid);
  assert(r == 1);

  if (nnz_row0 > 0) {
    A = s_toeplitz_create(N, nnz_row0);

    r = fread(A->v_row0, sizeof(elem), nnz_row0, fid);
    assert(r == nnz_row0);

    r = fread(A->j_row0, sizeof(int), nnz_row0, fid);
    assert(r == nnz_row0);
  }
  else if (nnz_row0 == 0) {
    return NULL;
  }
  else {
    assert(0);
    exit(0);
  }

  return A;
}


void s_toeplitz_export(const char *filename, const s_toeplitz *A) {
  FILE *fid;

  assert(filename);
  assert(A);

  fid = fopen(filename, "w");
  assert(fid);

  s_toeplitz_export_fid(fid, A);

  fclose(fid);
}


void s_toeplitz_export_fid(FILE *fid, const s_toeplitz *A) {
  int r;
  int sizeof_elem;

  assert(A);

  sizeof_elem = sizeof(elem);
  r = fwrite(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&(A->n), sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&(A->nnz_row0), sizeof(int), 1, fid);
  assert(r == 1);


  r = fwrite(A->v_row0, sizeof(elem), A->nnz_row0, fid);
  assert(r == A->nnz_row0);

  r = fwrite(A->j_row0, sizeof(int), A->nnz_row0, fid);
  assert(r == A->nnz_row0);
}

int s_toeplitz_get_nnz(s_toeplitz *A) {
  int i, nnz;

  assert(A);

  nnz = 0;
  for (i = 0; i < A->nnz_row0; i++) {
    if (A->j_row0[i] == 0) {
      nnz += A->n;
    }
    else {
      nnz += 2*(A->n - A->j_row0[i]);
    }
  }

  return nnz;
}



/******************************************************************************/

/*******************
 * s_block_toeplitz
 *******************/


s_block_toeplitz *s_block_toeplitz_create(int K) {
  s_block_toeplitz *A;

  assert(K > 0);

  A = malloc(sizeof(s_block_toeplitz));
  assert(A);

  A->n = -1;
  A->K = K;
  A->n_block = -1;

  A->B = malloc(sizeof(s_toeplitz *) * K);
  assert(A->B);

  A->row = -1;
  A->v = NULL;
  A->j = NULL;
  A->nnz = 0;

  A->max_row_length = -1;

  return A;
}


void s_block_toeplitz_destroy(s_block_toeplitz **A) {
  int k;

  assert(A);
  assert(*A);

  for (k = 0; k < (*A)->K; k++) {
    if ((*A)->B[k]->nnz_row0 > 0) {
      s_toeplitz_destroy(&((*A)->B[k]));
    }
    else if ((*A)->B[k]->nnz_row0 == 0) {
      free((*A)->B[k]);
    }
    else {
      assert(0);
    }
  }

  free((*A)->B);

  if ((*A)->v) {
    free((*A)->v);
  }

  if ((*A)->j) {
    free((*A)->j);
  }

  free(*A);
  *A = NULL;
}


s_block_toeplitz *s_block_toeplitz_import(const char *filename) {
  s_block_toeplitz *A;
  FILE *fid;
  int K, k, max_row_length;
  int r;
  int sizeof_elem;

  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  r = fread(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);
  assert(sizeof_elem == sizeof(elem));

  r = fread(&K, sizeof(int), 1, fid);
  assert(r == 1);

  A = s_block_toeplitz_create(K);
  A->n_block = -1;

  max_row_length = 0;
  for (k = 0; k < K; k++) {
    A->B[k] = s_toeplitz_import_fid(fid);

    if (A->B[k] == NULL) {
      /* Because it would cause problems in the next conditional when
	 k = 0. */
      assert(k != 0);

      A->B[k] = malloc(sizeof(s_toeplitz));
      assert(A->n_block > 0);
      A->B[k]->n = A->n_block;
      A->B[k]->v_row0 = NULL;
      A->B[k]->j_row0 = NULL;
      A->B[k]->nnz_row0 = 0;
      A->B[k]->row = -1;
      A->B[k]->v = NULL;
      A->B[k]->j = NULL;
      A->B[k]->nnz = 0;
    }
    else {
      if (k == 0) {
	max_row_length += 2*A->B[k]->nnz_row0 - 1;
      }
      else {
	max_row_length += 2*(2*A->B[k]->nnz_row0 - 1);
      }
    }

    if (k == 0) {
      A->n_block = A->B[0]->n;
    }
    else {
      assert(A->B[k]->n == A->n_block);
    }
  }

  fclose(fid);

  A->n = K*A->n_block;

  /* At most, a row of A can have max_row_length elements, where
     max_row_length = 2*nnz_row0[0] - 1 +
                      sum_{k=1}^{K-1} 2*(2*nnz_row0[k] - 1) */

  A->v = malloc(sizeof(elem) * max_row_length);
  assert(A->v);

  A->j = malloc(sizeof(int) * max_row_length);
  assert(A->j);

  A->max_row_length = max_row_length;

  return A;
}


void s_block_toeplitz_export(const char *filename, const s_block_toeplitz *A) {
  FILE *fid;
  int k;
  int r;
  int sizeof_elem;

  assert(filename);
  assert(A);

  fid = fopen(filename, "w");
  assert(fid);

  sizeof_elem = sizeof(elem);
  r = fwrite(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&(A->K), sizeof(int), 1, fid);
  assert(r == 1);

  for (k = 0; k < A->K; k++) {
    if (A->B[k]->nnz_row0 > 0) {
      s_toeplitz_export_fid(fid, A->B[k]);
    }
    else if (A->B[k]->nnz_row0 == 0) {
      r = fwrite(&A->B[k]->n, sizeof(int), 1, fid);
      assert(r == 1);

      r = fwrite(&A->B[k]->nnz_row0, sizeof(int), 1, fid);
      assert(r == 1);
    }
    else {
      assert(0);
    }
  }

  fclose(fid);
}


void s_block_toeplitz_printf_blocks(const s_block_toeplitz *A) {
  int k;

  for (k = 0; k < A->K; k++) {
    printf("k=%d:\n", k);
    if (A->B[k]->nnz_row0 > 0) {
      s_toeplitz_printf(A->B[k]);
    }
    else if (A->B[k]->nnz_row0 == 0) {
      printf("[Empty]\n");
    }
    else {
      assert(0);
    }
    printf("\n");

    if (k != A->K -1) {
      printf("\n");
    }
  }
}


void s_block_toeplitz_get_row(s_block_toeplitz *A, int row) {
  div_t d;
  int l, r, k, i;
  int index, block, shift;

  assert(A);
  assert(row >= 0 && row <= A->n);

  /* l is the row block */
  /* r is the row within each block */
  d = div(row, A->n_block);
  l = d.quot;
  r = d.rem;

  index = 0;
  for (k = 0; k < A->K; k++) {
    block = abs(k - l);
    assert(block < A->K);    /* block >= 0 */

    if (A->B[block]->nnz_row0 > 0) {
      s_toeplitz_get_row(A->B[block], r);

      assert(index + A->B[block]->nnz <= A->max_row_length);

      /* Copy v */
      memcpy(&(A->v[index]), A->B[block]->v, A->B[block]->nnz * sizeof(elem));

      /* Copy and shift j */
      shift = k * A->n_block;
      for (i = 0; i < A->B[block]->nnz; i++) {
	A->j[index + i] = A->B[block]->j[i] + shift;
      }

      index += A->B[block]->nnz;
    }
    else {
      assert(A->B[block]->nnz_row0 == 0);
    }
  }

  A->nnz = index;
}


void s_block_toeplitz_printf_row(s_block_toeplitz *A, int row) {
  int i, index;

  assert(A);
  assert(row >= 0 && row < A->n);

  s_block_toeplitz_get_row(A, row);

  index = 0;
  for (i = 0; i < A->n; i++) {
    if (index < A->nnz && i == A->j[index]) {
      printf_elem_s(A->v[index]);
      index++;
    }
    else {
      printf_elem_s((elem) 0);
    }
  }

  assert(index == A->nnz);
}


elem s_block_toeplitz_get(s_block_toeplitz *A, int i, int j) {
  int block_index;
  int i_rel, j_rel, column_index;
  int col;

  assert(A);
  assert(i >= 0 && i < A->n);
  assert(j >= 0 && j < A->n);

  block_index = fabs(floor(i / A->n_block) - floor(j / A->n_block));

  i_rel = i % A->n_block;
  j_rel = j % A->n_block;
  column_index = abs(i_rel - j_rel);

  if (A->B[block_index]->j_row0)
    if (A->B[block_index]->j_row0[column_index] == column_index) {
      return A->B[block_index]->v_row0[column_index];
    }
    else {
      for (col = 0; col < A->B[block_index]->nnz; col++) {
	if (A->B[block_index]->j_row0[col] == column_index) {
	  return A->B[block_index]->v_row0[col];
	}
      }

      return 0;
    }
  else {
    return 0;
  }
}


int s_block_toeplitz_get_nnz(s_block_toeplitz *A) {
  int i, nnz;

  assert(A);

  nnz = 0;

  if (A->B[0] && A->B[0]->nnz_row0 > 0) {
    nnz += A->K*s_toeplitz_get_nnz(A->B[0]);
  }

  for (i = 1; i < A->K; i++) {
    if (A->B[i] && A->B[i]->nnz_row0 > 0) {
      nnz += 2*(A->K - i)*s_toeplitz_get_nnz(A->B[i]);
    }
  }

  return nnz;
}


sparse_rcs *s_block_toeplitz_convert(s_block_toeplitz *A) {
  sparse_rcs *S;
  int i, l;
  int index;

  assert(A);

  S = sparse_rcs_create(A->n, A->n, s_block_toeplitz_get_nnz(A));

  index = 0;
  for (i = 0; i < A->n; i++) {
    s_block_toeplitz_get_row(A, i);

    S->r[i] = index;

    for (l = 0; l < A->nnz; l++) {
      S->v[index + l] = A->v[l];
      S->j[index + l] = A->j[l];
    }

    index += A->nnz;
  }

  S->r[A->n] = index;

  assert(index == S->N);

  return S;
}
