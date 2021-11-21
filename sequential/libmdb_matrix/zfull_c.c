#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "zfull_c.h"
#include "eblas.h"
#include "elem.h"


zfull_c *zfull_c_create(int m, int n) {
  zfull_c *A = NULL;
  int j;

  assert(m > 0);
  assert(n > 0);

  A = malloc(sizeof(zfull_c));
  assert(A);

  A->m = m;
  A->n = n;

  /* Fancy allocation scheme to guarantee the dynamically allocated
   * multidimensional array is contiguous in memory (see C FAQ 6.16) */

  A->v = malloc(sizeof(z_elem *) * n);
  A->v[0] = malloc(sizeof(z_elem) * m * n);
  for (j = 1; j < n; j++) {
    A->v[j] = A->v[0] + j * m;
  }

  A->v_vector = A->v[0];

  return A;
}


void zfull_c_destroy(zfull_c **A) {
  assert(A);
  assert(*A);
  assert((*A)->v);
  assert((*A)->v[0]);

  free((*A)->v[0]);
  free((*A)->v);
  free(*A);

  *A = NULL;
}

void zfull_c_printf(const zfull_c *A) {
  int i, j;

  assert(A);

  for (i = 0; i < A->m; i++) {
    for (j = 0; j < A->n; j++) {
      printf_z_elem_s((const z_elem *) &(A->v[j][i]));
    }
    printf("\n");
  }
}


void zfull_c_export(const char *filename, const zfull_c *A) {
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


void zfull_c_set0(zfull_c *A) {
  assert(A);

  set0(&(A->v[0][0][0]), 2 * A->m * A->n);
}
