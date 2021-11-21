#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "toeplitz_new.h"


/******************************************************************************/
/* s_toe */


s_toe *s_toe_create(int n, int nnz_row0) {
  s_toe *A;

  assert(n >= 0);
  assert(nnz_row0 >= 0);
  assert(nnz_row0 <= n);

  A = malloc(sizeof(s_toe));
  assert(A);

  A->n = n;
  A->nnz_row0 = nnz_row0;

  if (nnz_row0 == 0) {
    A->v_row0 = NULL;
  }
  else {
    A->v_row0 = malloc(nnz_row0 * sizeof(elem));
    assert(A->v_row0);
  }

  return A;
}


void s_toe_destroy(s_toe **A) {
  assert(A);
  assert(*A);

  if ((*A)->nnz_row0 == 0) {
    assert((*A)->v_row0 == NULL);
  }
  else {
    free((*A)->v_row0);
  }

  free(*A);

  *A = NULL;
}


void s_toe_printf(const s_toe *A) {
  int i;
  s_toe_it it;

  assert(A);

  for (i = 0; i < A->n; i++) {
    s_toe_iterator(A, &it, i);

    while (s_toe_it_has_next(&it)) {
      printf_elem_s(s_toe_it_next(&it));
    }
    printf("\n");
  }
}


s_toe *s_toe_import(const char *filename) {
  FILE *fid;
  s_toe *A;

  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  A = s_toe_import_fid(fid);

  fclose(fid);

  return A;
}


s_toe *s_toe_import_fid(FILE *fid) {
  s_toe *A;
  int n;
  int nnz_row0;
  int r;
  int sizeof_elem;

  assert(fid);

  r = fread(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);
  assert(sizeof_elem == sizeof(elem));

  r = fread(&n, sizeof(int), 1, fid);
  assert(r == 1);

  r = fread(&nnz_row0, sizeof(int), 1, fid);
  assert(r == 1);

  A = s_toe_create(n, nnz_row0);

  if (A->nnz_row0 > 0) {
    r = fread(A->v_row0, sizeof(elem), A->nnz_row0, fid);
    assert(r == A->nnz_row0);
  }

  return A;
}


void s_toe_export(const char *filename, const s_toe *A) {
  FILE *fid;

  assert(filename);
  assert(A);

  fid = fopen(filename, "w");
  assert(fid);

  s_toe_export_fid(fid, A);

  fclose(fid);
}


void s_toe_export_fid(FILE *fid, const s_toe *A) {
  int r;
  int sizeof_elem;

  assert(fid);
  assert(A);

  sizeof_elem = sizeof(elem);
  r = fwrite(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&A->n, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&A->nnz_row0, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(A->v_row0, sizeof(elem), A->nnz_row0, fid);
  assert(r == A->nnz_row0);
}


elem s_toe_get(const s_toe *A, int i, int j) {
  int k;

  assert(A);
  assert(i >= 0);
  assert(i < A->n);
  assert(j >= 0);
  assert(j < A->n);

  k = abs(i - j);

  if (k < A->nnz_row0) {
    return (A->v_row0[k]);
  }
  else {
    return 0;
  }
}


/******************************************************************************/

void s_toe_iterator(const s_toe *A, s_toe_it *it, int i) {
  assert(A);
  assert(it);
  assert(i >= 0);
  assert(i < A->n);

  it->A = A;
  it->i = i;
  it->j = -1;
}

boolean s_toe_nz_next_check(const s_toe *A, int i, int j) {
  assert(A);

  if (j == -1) {
    return (A->nnz_row0 != 0);
  }
  else {
    if (j >= i) {
      return ((j - i < A->nnz_row0 - 1) &&
	      (j < A->n - 1));
    }
    else {
      return True;
    }
  }

}

boolean s_toe_nz_it_has_next(const s_toe_it *it) {
  assert(it);

  return s_toe_nz_next_check(it->A, it->i, it->j);

}

elem s_toe_nz_it_next(s_toe_it *it) {
  assert(s_toe_nz_it_has_next(it));

  if (it->j == -1) {
    if (it->i >= it->A->nnz_row0) {
      it->j = it->i - it->A->nnz_row0 + 1;
    }
    else {
      it->j = 0;
    }
  }
  else {
    it->j++;
  }

  assert(abs(it->i - it->j) < it->A->nnz_row0);
  it->v = it->A->v_row0[abs(it->i - it->j)];

  return it->v;
}


boolean s_toe_next_check(const s_toe *A, int j) {
  assert(A);

  if (j == -1) {
    return True;
  }
  else {
    assert(j >= 0);
    return (j < A->n - 1);
  }
}

boolean s_toe_it_has_next(const s_toe_it *it) {
  assert(it);

  return s_toe_next_check(it->A, it->j);
}

elem s_toe_it_next(s_toe_it *it) {
  int k;

  assert(s_toe_it_has_next(it));

  it->j++;

  k = abs(it->i - it->j);

  if (k < it->A->nnz_row0) {
    it->v = it->A->v_row0[k];
  }
  else {
    it->v = 0;
  }

  return it->v;
}



/******************************************************************************/
/* sb_toe */

sb_toe *sb_toe_create(int n_block, int k, int k_nnz, vector *nnz_row0_list) {
  sb_toe *A;
  int i;

  assert(n_block >= 0);
  assert(k >= 0);
  assert(k_nnz >= 0);
  assert(k_nnz <= k);
  assert(nnz_row0_list);
  assert(nnz_row0_list->n == k_nnz);

  A = malloc(sizeof(sb_toe));
  assert(A);

  A->n = n_block * k;
  A->n_block = n_block;
  A->k = k;
  A->k_nnz = k_nnz;

  A->A_block = malloc(sizeof(s_toe) * k);
  assert(A->A_block);

  for (i = 0; i < k; k++) {
    A->A_block[i] = s_toe_create(n_block, nnz_row0_list->v[i]);
  }

  return A;
}


void sb_toe_destroy(sb_toe **A) {
  int i;

  assert(A);
  assert(*A);

  for (i = 0; i < (*A)->k; i++) {
    s_toe_destroy(&((*A)->A_block[i]));
  }

  free((*A)->A_block);
  free(*A);

  *A = NULL;
}


sb_toe *sb_toe_import(const char *filename) {
  FILE *fid;
  sb_toe *A;

  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  A = sb_toe_import_fid(fid);

  fclose(fid);

  return A;
}


sb_toe *sb_toe_import_fid(FILE *fid) {
  sb_toe *A;
  int n_block;
  int k;
  int k_nnz;
  int i;
  int r;
  int sizeof_elem;

  assert(fid);

  r = fread(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);
  assert(sizeof_elem == sizeof(elem));

  r = fread(&k, sizeof(int), 1, fid);
  assert(r == 1);

  r = fread(&n_block, sizeof(int), 1, fid);
  assert(r == 1);

  r = fread(&k_nnz, sizeof(int), 1, fid);
  assert(r == 1);

  assert(n_block >= 0);
  assert(k >= 0);
  assert(k_nnz >= 0);

  A = malloc(sizeof(sb_toe));
  assert(A);

  A->n = n_block * k;
  A->n_block = n_block;
  A->k = k;
  A->k_nnz = k_nnz;

  A->A_block = malloc(sizeof(s_toe *) * k);
  assert(A->A_block);

  for (i = 0; i < k; i++) {
    A->A_block[i] = s_toe_import_fid(fid);
  }

  return A;
}


void sb_toe_export(const char *filename, const sb_toe *A) {
  FILE *fid;

  assert(filename);
  assert(A);

  fid = fopen(filename, "w");
  assert(fid);

  sb_toe_export_fid(fid, A);

  fclose(fid);
}


void sb_toe_export_fid(FILE *fid, const sb_toe *A) {
  int i;
  int r;
  int sizeof_elem = 0;

  assert(fid);
  assert(A);

  r = fwrite(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);
  assert(sizeof_elem == sizeof(elem));

  r = fwrite(&A->k, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&A->n_block, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&A->k_nnz, sizeof(int), 1, fid);
  assert(r == 1);

  for (i = 0; i < A->k; i++) {
    s_toe_export_fid(fid, A->A_block[i]);
  }
}


void sb_toe_printf(const sb_toe *A) {
  int i;
  sb_toe_it it;

  assert(A);

  for (i = 0; i < A->n; i++) {
    sb_toe_iterator(A, &it, i);

    while (sb_toe_it_has_next(&it)) {
      printf_elem_s(sb_toe_it_next(&it));
    }
    printf("\n");
  }
}


/******************************************************************************/

void sb_toe_iterator(const sb_toe *A, sb_toe_it *it, int i) {
  div_t result;

  assert(A);
  assert(it);
  assert(i >= 0);
  assert(i < A->n);

  it->A = A;
  it->i = i;
  it->j = -1;

  result = div(it->i, it->A->n_block);
  it->block_i = result.quot;

  it->block_j = -1;
  it->block = -1;
}


boolean sb_toe_nz_next_check(const sb_toe *A, int block_i, int block_j,
			     const s_toe_it *block_it) {
  assert(A);

  if (block_j == -1) {
    return (A->k_nnz != 0);
  }
  else if (block_j < block_i) {
    /* Case: left of diagonal */
    return True;
  }
  else if (block_j == A->k - 1) {
    /* Case: block is at right edge */
    return s_toe_nz_it_has_next(block_it);
  }
  else if (abs(block_i - block_j) >= A->k_nnz -1) {
    /* Case: right of diagonal */
    return s_toe_nz_it_has_next(block_it);
  }
  else {
    /* Case: on diagonal, but not at edge */
    assert(block_i == block_j);
    return True;
  }
}


boolean sb_toe_nz_it_has_next(const sb_toe_it *it) {
  assert(it);

  return sb_toe_nz_next_check(it->A, it->block_i, it->block_j, &(it->block_it));
}


elem sb_toe_nz_it_next(sb_toe_it *it) {
  assert(it);
  assert(sb_toe_nz_it_has_next(it));

  if (it->block_j == -1) {
    int local_i;

    local_i = it->i % it->A->n_block;

    if (it->block_i >= it->A->k_nnz) {
      it->block_j = it->block_i - it->A->k_nnz + 1;
    }
    else {
      it->block_j = 0;
    }

    it->block = abs(it->block_i - it->block_j);
    s_toe_iterator(it->A->A_block[it->block], &(it->block_it), local_i);

    it->v = s_toe_nz_it_next(&(it->block_it));
    it->j = it->block_j * it->A->n_block + it->block_it.j;
  }
  else {
    if (s_toe_nz_it_has_next(&(it->block_it))) {
      it->v = s_toe_nz_it_next(&(it->block_it));
      it->j++;
    }
    else {
      it->block_j++;

      it->block = abs(it->block_i - it->block_j);
      s_toe_iterator(it->A->A_block[it->block], &(it->block_it),
		     it->block_it.i);

      it->v = s_toe_nz_it_next(&(it->block_it));
      it->j = it->block_j * it->A->n_block + it->block_it.j;
    }
  }

  return it->v;
}


boolean sb_toe_next_check(const sb_toe *A, int block_j,
			  const s_toe_it *block_it) {
  assert(A);
  assert(block_j < A->k);

  if (block_j < A->k - 1) {
    return True;
  }
  else {
    return s_toe_next_check(block_it->A, block_it->j);
  }
}


boolean sb_toe_it_has_next(const sb_toe_it *it) {
  assert(it);

  return sb_toe_next_check(it->A, it->block_j, &(it->block_it));
}


elem sb_toe_it_next(sb_toe_it *it) {
  assert(it);
  assert(sb_toe_it_has_next(it));

  if (it->block_j == -1) {
    int local_i;

    local_i = it->i % it->A->n_block;

    it->block_j = 0;
    it->block = abs(it->block_i - it->block_j);
    s_toe_iterator(it->A->A_block[it->block], &(it->block_it), local_i);

    it->v = s_toe_it_next(&(it->block_it));
    it->j = it->block_it.j;
  }
  else {
    if (s_toe_it_has_next(&(it->block_it))) {
      it->v = s_toe_it_next(&(it->block_it));
      it->j++;
    }
    else {
      it->block_j++;

      it->block = abs(it->block_i - it->block_j);
      s_toe_iterator(it->A->A_block[it->block], &(it->block_it),
		     it->block_it.i);

      it->v = s_toe_it_next(&(it->block_it));
      it->j = it->block_j * it->A->n_block + it->block_it.j;
    }
  }

  return it->v;
}



/******************************************************************************/
/* sb_toe_r */

static void sb_toe_r_create_r(sb_toe_r *A, int depth);
static void sb_toe_r_destroy_r(sb_toe_r *A, int depth);
static void sb_toe_r_import_r(const sb_toe_r *A, FILE *fid, int depth);
static void sb_toe_r_export_r(const sb_toe_r *A, FILE *fid, int depth);

static elem sb_toe_r_nz_it_move_right(sb_toe_r_it *A_it);


sb_toe_r *sb_toe_r_create(int rank, int *n_phy, int *n) {
  sb_toe_r *A;
  sb_toe_r_dim *dim;
  int i;

  assert(rank > 0);
  assert(n_phy);
  assert(n);

  for (i = 0; i < rank; i++) {
    assert(n_phy[i] >= 0);
    assert(n[i] > 0);
    assert(n[i] >= n_phy[i]);
  }

  A = malloc(sizeof(sb_toe_r));
  assert(A);

  dim = malloc(sizeof(sb_toe_r_dim));
  assert(dim);

  dim->rank = rank;
  dim->n_phy = n_phy;
  dim->n = n;

  dim->N_phy = malloc(rank * sizeof(int));
  assert(dim->N_phy);

  dim->N = malloc(rank * sizeof(int));
  assert(dim->N);

  dim->N_phy[rank-1] = dim->n_phy[rank-1];
  dim->N[rank-1] = dim->n[rank-1];
  for (i = rank-2; i >= 0; i--) {
    dim->N_phy[i] = dim->n_phy[i] * dim->N_phy[i+1];
    dim->N[i] = dim->n[i] * dim->N[i+1];
  }

  A->dim = dim;

  sb_toe_r_create_r(A, 0);

  return A;
}


static void sb_toe_r_create_r(sb_toe_r *A, int depth) {
  int i;

  assert(A);
  assert(depth >= 0);
  assert(depth < A->dim->rank);

  if (depth == A->dim->rank - 1) {
    A->ptr.v = malloc(A->dim->n_phy[depth] * sizeof(elem));
    assert(A->ptr.v);
  }
  else {
    A->ptr.A_block = malloc(A->dim->n_phy[depth] * sizeof(sb_toe_r *));
    assert(A->ptr.A_block);

    for (i = 0; i < A->dim->n_phy[depth]; i++) {
      A->ptr.A_block[i] = malloc(sizeof(sb_toe_r));
      assert(A->ptr.A_block[i]);

      A->ptr.A_block[i]->dim = A->dim;
      sb_toe_r_create_r(A->ptr.A_block[i], depth+1);
    }
  }
}

void sb_toe_r_destroy(sb_toe_r **A) {
  assert(A);
  assert(*A);

  sb_toe_r_destroy_r(*A, 0);
  *A = NULL;
}


void sb_toe_r_destroy_r(sb_toe_r *A, int depth) {
  int i;

  assert(A);
  assert(depth >= 0);
  assert(depth < A->dim->rank);

  if (depth == A->dim->rank - 1) {
    free(A->ptr.v);
  }
  else {
    for (i = 0; i < A->dim->n_phy[depth]; i++) {
      sb_toe_r_destroy_r(A->ptr.A_block[i], depth + 1);
    }

    free(A->ptr.A_block);
  }

  if (depth == 0) {
    free(A->dim->n_phy);
    free(A->dim->n);
    free(A->dim->N_phy);
    free(A->dim->N);
    free(A->dim);
  }

  free(A);
}


void sb_toe_r_printf(const sb_toe_r *A) {
  sb_toe_r_fprintf(stdout, A);
}


void sb_toe_r_fprintf(FILE *fid, const sb_toe_r *A) {
  int i;
  sb_toe_r_it *A_it;
  elem e;

  assert(fid);
  assert(A);


  A_it = sb_toe_r_it_create(A);

  for (i = 0; i < A->dim->N[0]; i++) {
    sb_toe_r_it_init(A_it, i);

    while(sb_toe_r_it_has_next(A_it)) {
      e = sb_toe_r_it_next(A_it);
      fprintf_elem_s(fid, e);
    }
    fprintf(fid, "\n");
  }

  sb_toe_r_it_destroy(&A_it);
}


sb_toe_r *sb_toe_r_import(const char *filename) {
  FILE *fid;
  sb_toe_r *A;
  int rank;
  int *n_phy;
  int *n;
  int r;
  int sizeof_elem;

  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  r = fread(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);
  assert(sizeof_elem == sizeof(elem));

  r = fread(&rank, sizeof(int), 1, fid);
  assert(r == 1);

  assert(rank > 0);

  n_phy = malloc(rank * sizeof(int));
  assert(n_phy);

  n = malloc(rank * sizeof(int));
  assert(n);

  r = fread(n_phy, sizeof(int), rank, fid);
  assert(r == rank);

  r = fread(n, sizeof(int), rank, fid);
  assert(r == rank);

  A = sb_toe_r_create(rank, n_phy, n);

  sb_toe_r_import_r(A, fid, 0);

  fclose(fid);

  return A;
}


static void sb_toe_r_import_r(const sb_toe_r *A, FILE *fid, int depth) {
  int i;
  int r;

  assert(A);
  assert(fid);
  assert(depth >= 0);
  assert(depth < A->dim->rank);

  if (depth == A->dim->rank - 1) {
    r = fread(A->ptr.v, sizeof(elem), A->dim->n_phy[depth], fid);
    assert(r ==  A->dim->n_phy[depth]);
  }
  else {
    for (i = 0; i < A->dim->n_phy[depth]; i++) {
      sb_toe_r_import_r(A->ptr.A_block[i], fid, depth+1);
    }
  }
}


void sb_toe_r_export(const char *filename, const sb_toe_r *A) {
  FILE *fid;
  int sizeof_elem;
  int rank;
  int r;

  assert(filename);
  assert(A);

  fid = fopen(filename, "w");
  assert(fid);

  sizeof_elem = sizeof(elem);
  r = fwrite(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);

  rank = A->dim->rank;

  r = fwrite(&rank, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(A->dim->n_phy, sizeof(int), rank, fid);
  assert(r == rank);

  r = fwrite(A->dim->n, sizeof(int), rank, fid);
  assert(r == rank);

  sb_toe_r_export_r(A, fid, 0);

  fclose(fid);
}


void sb_toe_r_export_r(const sb_toe_r *A, FILE *fid, int depth) {
  sb_toe_r_dim *dim;
  int rank;

  int i, r;

  assert(fid);
  assert(A);

  dim = A->dim;
  rank = dim->rank;

  assert(depth >= 0);
  assert(depth < rank);


  if (depth == rank - 1) {
    r = fwrite(A->ptr.v, sizeof(elem), dim->n_phy[depth], fid);
    assert(r == dim->n_phy[depth]);
  }
  else {
    for (i = 0; i < dim->n_phy[depth]; i++) {
      sb_toe_r_export_r(A->ptr.A_block[i], fid, depth+1);
    }
  }
}


void sb_toe_r_it_init(sb_toe_r_it *A_it, int i) {
  int rank;
  int k, i_div, i_mod;
  const sb_toe_r *A;

  assert(A_it->A);

  A = A_it->A;

  assert(i >= 0);
  assert(i < A->dim->N[0]);
  assert(A_it);

  rank = A->dim->rank;

  A_it->A = A;
  A_it->i = i;
  A_it->j = -1;
  A_it->j_last = A->dim->N[0];

  assert(A_it->j_state);
  assert(A_it->j_state->n == rank);
  assert(A_it->j_state->max == A->dim->n);

  counter_reset(A_it->j_state);
  A_it->j_state->c[rank - 1] = -1;

  assert(A_it->i_state);


  i_mod = i;
  for (k = 0; k < rank - 1; k++) {
    i_div = i_mod / A->dim->N[k+1];
    i_mod %= A->dim->N[k+1];

    A_it->i_state[k] = i_div;
  }

  A_it->i_state[rank - 1] = i_mod;
}


sb_toe_r_it *sb_toe_r_it_create(const sb_toe_r *A) {
  sb_toe_r_it *A_it;
  int rank;

  assert(A);

  rank = A->dim->rank;

  A_it = malloc(sizeof(sb_toe_r_it));
  assert(A_it);

  A_it->A = A;

  A_it->j_state = counter_create(rank, A->dim->n);

  A_it->i_state = malloc(rank * sizeof(int));

  return A_it;
}


void sb_toe_r_it_destroy(sb_toe_r_it **A_it) {
  assert(A_it);
  assert(*A_it);

  counter_destroy_keep_max(&(*A_it)->j_state);
  free((*A_it)->i_state);

  free(*A_it);
  *A_it = NULL;
}


boolean sb_toe_r_it_has_next(const sb_toe_r_it *A_it) {
  assert(A_it);

  return (A_it->j + 1 < A_it->j_last);
}


elem sb_toe_r_it_next(sb_toe_r_it *A_it) {
  int i;
  const sb_toe_r *A_ptr;
  int rank;
  int k;

  assert(A_it);
  assert(sb_toe_r_it_has_next(A_it));

  rank = A_it->A->dim->rank;

  A_it->j++;
  counter_tick(A_it->j_state);

  A_ptr = A_it->A;
  for (i = 0; i < rank - 1; i++) {
    k = abs(A_it->j_state->c[i] - A_it->i_state[i]);
    if (k >= A_it->A->dim->n_phy[i]) {
      return 0;
    }

    A_ptr = A_ptr->ptr.A_block[k];
  }

  k = abs(A_it->j_state->c[rank-1] - A_it->i_state[rank-1]);
  if (k >= A_it->A->dim->n_phy[rank - 1]) {
    return 0;
  }
  else {
    return A_ptr->ptr.v[k];
  }
}


sb_toe_r_it *sb_toe_r_nz_it_create(const sb_toe_r *A) {
  return sb_toe_r_it_create(A);
}


void sb_toe_r_nz_it_destroy(sb_toe_r_it **A_it) {
  sb_toe_r_it_destroy(A_it);
}


static elem sb_toe_r_nz_it_move_right(sb_toe_r_it *A_it) {
  int i;
  const sb_toe_r *A_ptr;
  const sb_toe_r_dim *dim;
  int rank;
  int k;

  assert(A_it);
  assert(sb_toe_r_it_has_next(A_it));

  dim = A_it->A->dim;
  rank = dim->rank;

  A_it->j++;
  counter_tick(A_it->j_state);

  for (i = 0; i < rank; i++) {
    k = A_it->j_state->c[i] - A_it->i_state[i];

    while (k <= -dim->n_phy[i]) {
      /* On the left of nonzero bands */
      if (i == rank - 1) {
	 A_it->j++;
	 counter_tick(A_it->j_state);
      }
      else {
	A_it->j += dim->N[i+1];
	counter_tick_digit_reset(A_it->j_state, i);
      }

      k = A_it->j_state->c[i] - A_it->i_state[i];
    }


    if (k >= dim->n_phy[i]) {
      /* On the right of nonzero bands */
      if (i == rank - 1) {
	A_it->j += dim->n[i] - A_it->j_state->c[i];
      }
      else {
	A_it->j += (dim->n[i] - A_it->j_state->c[i]) * dim->N[i+1];
      }

      assert(i-1 >= 0);
      counter_tick_digit_reset(A_it->j_state, i-1);

      /* Subtracting 2 because i will be incremented at the end of
	 the loop */
      i -= 2;
      continue;
    }
    else {
      /* Within nonzero bands */
      continue;
    }
  }

  A_ptr = A_it->A;
  for (i = 0; i < rank - 1; i++) {
    k = abs(A_it->j_state->c[i] - A_it->i_state[i]);
    assert(k < dim->n_phy[i]);

    A_ptr = A_ptr->ptr.A_block[k];
  }

  k = abs(A_it->j_state->c[rank-1] - A_it->i_state[rank-1]);
  assert(k < dim->n_phy[rank - 1]);

  return A_ptr->ptr.v[k];
}


void sb_toe_r_nz_it_init(sb_toe_r_it *A_it, int i) {
  int j_last;

  sb_toe_r_it_init(A_it, A_it->A->dim->N[0] - i - 1);
  sb_toe_r_nz_it_move_right(A_it);
  j_last = A_it->A->dim->N[0] - A_it->j;

  sb_toe_r_it_init(A_it, i);
  sb_toe_r_nz_it_move_right(A_it);

  A_it->j--;
  A_it->j_state->c[A_it->A->dim->rank - 1]--;
  A_it->j_last = j_last;
}


boolean sb_toe_r_nz_it_has_next(const sb_toe_r_it *A_it) {
  return sb_toe_r_it_has_next(A_it);
}


elem sb_toe_r_nz_it_next(sb_toe_r_it *A_it) {
  elem v;

  assert(A_it);
  assert(sb_toe_r_nz_it_has_next(A_it));

  v = sb_toe_r_nz_it_move_right(A_it);

  return v;
}


int sb_toe_r_nnz(const sb_toe_r *A) {
  int nnz;
  int i;

  assert(A);

  nnz = 1;

  for (i = A->dim->rank - 1; i >= 0; i--) {
    nnz *=
      A->dim->n[i] + (A->dim->n_phy[i] - 1)*(2*A->dim->n[i] - A->dim->n_phy[i]);
  }

  return nnz;
}


sparse_coo *sb_toe_r_convert_coo(const sb_toe_r *A) {
  int N;
  sparse_coo *B;
  int i, idx;
  sb_toe_r_it *A_it;

  assert(A);

  N = sb_toe_r_nnz(A);

  B = sparse_coo_create(A->dim->N[0], A->dim->N[0], N);

  idx = 0;
  A_it = sb_toe_r_nz_it_create(A);
  for (i = 0; i < A->dim->N[0]; i++) {
    sb_toe_r_nz_it_init(A_it, i);

    while (sb_toe_r_nz_it_has_next(A_it)) {
      B->v[idx] = sb_toe_r_nz_it_next(A_it);
      B->i[idx] = i;
      B->j[idx] = A_it->j;
      idx++;
    }
  }
  sb_toe_r_nz_it_destroy(&A_it);

  assert(idx == N);

  return B;
}
