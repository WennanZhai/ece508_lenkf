#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "zfull_r.h"
#include "eblas.h"


zfull_r *zfull_r_create(int m, int n) {
  zfull_r *A = NULL;
  int i;

  assert(m > 0);
  assert(n > 0);

  A = malloc(sizeof(zfull_r));
  assert(A);

  A->m = m;
  A->n = n;

  /* Fancy allocation scheme to guarantee the dynamically allocated
   * multidimensional array is contiguous in memory (see C FAQ 6.16) */

  A->v = malloc(sizeof(z_elem *) * m);
  A->v[0] = malloc(sizeof(z_elem) * m * n);
  for (i = 1; i < m; i++) {
    A->v[i] = A->v[0] + i * n;
  }

  A->v_vector = A->v[0];

  return A;
}


void zfull_r_destroy(zfull_r **A) {
  assert(A);
  assert(*A);
  assert((*A)->v);
  assert((*A)->v[0]);

  free((*A)->v[0]);
  free((*A)->v);
  free(*A);

  *A = NULL;
}


/* C <- A * B */
void zfull_r_mmm(const zfull_r *A, const zfull_r *B, zfull_r *C) {
  z_elem alpha, beta;

  assert(A);
  assert(B);
  assert(C);
  assert(C->m == A->m);
  assert(C->n == B->n);
  assert(A->n == B->m);

  alpha[0] = 1;
  alpha[1] = 0;

  beta[0] = 1;
  beta[1] = 0;

  zegemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C->m, C->n, A->n,
	 (const z_elem *) &alpha,
	 (const z_elem *) A->v_vector, A->n,
	 (const z_elem *) B->v_vector, B->n,
	 (const z_elem *) &beta,
	 C->v_vector, C->n);
}


void zfull_r_r_foreach(const zfull_r *A, int i, void (*func)(const z_elem *)) {
  int j;

  assert(A);
  assert(func);
  assert(i >= 0 && i < A->m);

  for (j = 0; j < A->n; j++) {
    func((const z_elem *) &(A->v[i][j]));
  }
}


void zfull_r_printf(const zfull_r *A) {
  int i;

  assert(A);

  for (i = 0; i < A->m; i++) {
    zfull_r_r_foreach(A, i, printf_z_elem_s);
    printf("\n");
  }
}


void zfull_r_export(const char *filename, const zfull_r *A) {
  FILE *fid;
  int i;
  elem *e_ptr;
  int r;
  int sizeof_elem;

  assert(filename);
  assert(A);

  fid = fopen(filename, "w");
  assert(fid);

  sizeof_elem = sizeof(elem);
  r = fwrite(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&A->m, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&A->n, sizeof(int), 1, fid);
  assert(r == 1);

  /* Output real part */
  e_ptr = &(A->v_vector[0][0]);
  for (i = 0; i < A->m * A->n; i++) {
    r = fwrite(e_ptr, sizeof(elem), 1, fid);
    assert(r == 1);
    e_ptr += 2;
  }

  /* Output imag part */
  e_ptr = &(A->v_vector[0][1]);
  for (i = 0; i < A->m * A->n; i++) {
    r = fwrite(e_ptr, sizeof(elem), 1, fid);
    assert(r == 1);
    e_ptr += 2;
  }

  fclose(fid);
}


void zfull_r_set0(zfull_r *A) {
  assert(A);

  set0(&(A->v[0][0][0]), 2 * A->m * A->n);
}
