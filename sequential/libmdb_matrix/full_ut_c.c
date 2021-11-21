#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "full_ut_c.h"
#include "util.h"


#ifndef NDEBUG
static int ut_c_check_index(const ut_c *A, int i, int j);
#endif
static int ut_c_get_index(const ut_c *A, int i, int j);


ut_c *ut_c_create(int m, int n) {
  ut_c *A;
  int num_elem, d;

  assert(m >= 0 && n >= 0);

  A = malloc(sizeof(ut_c));
  assert(A);

  A->m = m;
  A->n = n;

  if (m >= n) {
    num_elem = n*n - n*(n-1)/2;
  }
  else {
    d = n - m;
    num_elem = n*n - n*(n-1)/2 - (d*d - d*(d-1)/2);
  }

  A->v = malloc(sizeof(elem)*num_elem);
  assert(A->v);
  A->N = num_elem;

  return A;
}

void ut_c_destroy(ut_c **A) {
  assert(*A);

  free((*A)->v);
  free(*A);

  *A = NULL;
}

ut_c *ut_c_import(const char *filename) {
  ut_c *A;
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

  A = ut_c_create(m, n);

  r = fread(A->v, sizeof(elem), A->N, fid);
  assert(r == A->N);

  fclose(fid);

  return A;
}

elem ut_c_get(const ut_c *A, int i, int j) {
  int index;

  assert(A);
  assert(ut_c_check_index(A, i, j));

  index = ut_c_get_index(A, i, j);

  return A->v[index];
}


#ifndef NDEBUG
static int ut_c_check_index(const ut_c *A, int i, int j) {
  return(j >= i &&
         i >= 0 && i < A->m &&
         j >= 0 && j < A->n);
}
#endif

static int ut_c_get_index(const ut_c *A, int i, int j) {
  assert(A);
  assert(ut_c_check_index(A, i, j));

  if (A->m >= A->n) {
    return j*(j+1)/2 + i;
  }
  else {
    if (j < A->m) {
      return j*(j+1)/2 + i;
    }
    else {
      return A->m*(A->m+1)/2 + (j-A->m)*A->m + i;
    }
  }
}

void ut_c_set(ut_c *A, int i, int j, elem v) {
  int index;

  assert(A);
  assert(ut_c_check_index(A, i, j));

  index = ut_c_get_index(A, i, j);

  A->v[index] = v;
}

void ut_c_foreach(const ut_c *A, void (*func)(elem)) {
  int i;

  for (i = 0; i < A->N; i++) {
    func(A->v[i]);
  }
}

void ut_c_r_foreach(const ut_c *A, int i, void (*func)(elem)) {
  int j;

  assert(A);
  assert(i >= 0 && i < A->m);

  for (j = i; j < A->n; j++) {
    func(ut_c_get(A, i, j));
  }
}

void ut_c_r_printf(const ut_c *A, int i) {
  int j;

  assert(A);
  assert(i >= 0 && i < A->m);

  for (j = 0; j < A->n; j++) {
    if (j < i) {
      printf("%+f ", 0.0);
    }
    else {
      printf("%+f ", ut_c_get(A, i, j));
    }
  }
}

void ut_c_printf(const ut_c *A) {
  int i;

  assert(A);

  for (i = 0; i < A->m; i++) {
    ut_c_r_printf(A, i);
    printf("\n");
  }
}

void full_c_2_ut_c(const full_c *A, ut_c *B) {
  int i, num_elem;
  elem *A_ptr, *B_ptr;

  assert(A);
  assert(B);
  assert((A->m == B->m) && (A->n == B->n));

  B_ptr = &(B->v[0]);

  for (i = 0; i < A->n; i++) {
    /* v[column][row] for full_c */
    A_ptr = &(A->v[i][0]);
    num_elem = MIN(i+1, A->m);
    memcpy(B_ptr, A_ptr, num_elem*sizeof(elem));
    B_ptr += num_elem;
  }
}

/*
void full_c_times_ut_c(full_c *A, const ut_c *B) {

}
*/
