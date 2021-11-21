#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "vector.h"
#include "eblas.h"


/******************************************************************************/

/*********
 * vector
 *********/

vector *vector_create(int n) {
  vector *v;

  assert(n > 0);


  v = malloc(sizeof(vector));
  assert(v);

  v->n = n;
  v->v = malloc(sizeof(elem) * n);
  assert(v->v);

  return v;
}


void vector_destroy(vector **v) {
  assert(*v);

  free((*v)->v);
  free(*v);
  *v = NULL;
}


void vector_fnprintf(FILE *fid, const vector *v, int n) {
  int i;

  assert(fid);
  assert(v);
  assert(n >= 0 && n <= v->n);

  for (i = 0; i < n; i++) {
    fprintf_elem_s(fid, v->v[i]);
  }
  fprintf(fid, "\n");
}


void vector_fprintf(FILE *fid, const vector *v) {
  vector_fnprintf(fid, v, v->n);
}


void vector_printf(const vector *v) {
  vector_fprintf(stdout, v);
}


vector *vector_import(const char *filename) {
  FILE *fid;
  int n, count;
  vector *v;
  int sizeof_elem;

  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  count = fread(&sizeof_elem, sizeof(int), 1, fid);
  assert(count == 1);
  assert(sizeof_elem == sizeof(elem));

  count = fread(&n, sizeof(int), 1, fid);
  assert(count == 1);

  v = vector_create(n);

  count = fread(v->v, sizeof(elem), n, fid);
  assert(count == n);

  fclose(fid);

  return v;
}


void vector_set0(vector *v) {
  assert(v);

  set0(v->v, v->n);
}


void vector_set_nan(vector *v) {
  elem nan;

  assert(v);

  nan = ENAN;
  ecopy(v->n, &nan, 0, v->v, 1);
}


void vector_export(const char *filename, const vector *v) {
  FILE *fid;
  int n;
  int sizeof_elem;

  assert(filename);
  assert(v);

  fid = fopen(filename, "w");
  assert(fid);

  sizeof_elem = sizeof(elem);
  n = fwrite(&sizeof_elem, sizeof(int), 1, fid);
  assert(n == 1);

  n = fwrite(&(v->n), sizeof(int), 1, fid);
  assert(n == 1);

  n = fwrite(v->v, sizeof(elem), v->n, fid);
  assert(n == v->n);

  fclose(fid);
}


/* y <- x */
void vector_copy(vector *y, const vector *x) {
  assert(y);
  assert(x);
  assert(y->n == x->n);

  ecopy(x->n, x->v, 1, y->v, 1);
}


/* f <- sqrt(x' * x) */
elem vector_norm(const vector *x) {
  assert(x);

  return enrm2(x->n, x->v, 1);
}


/* f <- x' * x */
elem vector_squared_norm(const vector *x) {
  assert(x);

  return edot(x->n, x->v, 1, x->v, 1);
}


/* x[i] <- x[i] + alpha */
void vector_add_scalar(vector *x, elem alpha) {
  elem one = 1;

  assert(x);

  eaxpy(x->n, alpha, &one, 0, x->v, 1);
}


/* x[i] <- alpha * x[i] */
void vector_scal(vector *x, elem alpha) {
  assert(x);

  escal(x->n, alpha, x->v, 1);
}



/******************************************************************************/

/**********
 * zvector
 **********/

zvector *zvector_create(int n) {
  zvector *v;

  assert(n > 0);


  v = malloc(sizeof(zvector));
  assert(v);

  v->n = n;
  v->v = malloc(sizeof(z_elem) * n);
  assert(v->v);

  return v;
}

void zvector_destroy(zvector **v) {
  assert(*v);

  free((*v)->v);
  free(*v);
  *v = NULL;
}

void zvector_export(const char *filename, const zvector *v) {
  FILE *fid;
  int r;
  int sizeof_elem;

  assert(filename);
  assert(v);

  fid = fopen(filename, "w");
  assert(fid);

  sizeof_elem = sizeof(elem);
  r = fwrite(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&(v->n), sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(v->v, sizeof(z_elem), v->n, fid);
  assert(r == v->n);

  fclose(fid);
}

void zvector_printf(const zvector *v) {
  int i;

  assert(v);

  for (i = 0; i < v->n; i++) {
    printf_z_elem_s((const z_elem *) v->v[i]);
  }
  printf("\n");
}

void zvector_set0(const zvector *v) {
  assert(v);

  zset0(v->v, v->n);
}


/******************************************************************************/

/**********
 * ivector
 **********/

ivector *ivector_create(int n) {
  ivector *v;

  assert(n > 0);


  v = malloc(sizeof(ivector));
  assert(v);

  v->n = n;
  v->v = malloc(sizeof(int) * n);
  assert(v->v);

  return v;
}

void ivector_destroy(ivector **v) {
  assert(*v);

  free((*v)->v);
  free(*v);
  *v = NULL;
}

void ivector_printf(const ivector *v) {
  int i;

  assert(v);

  for (i = 0; i < v->n; i++) {
    printf("%d ", v->v[i]);
  }
  printf("\n");
}

ivector *ivector_import(const char *filename) {
  FILE *fid;
  int n, count;
  ivector *v;
  int sizeof_int;

  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  count = fread(&sizeof_int, sizeof(int), 1, fid);
  assert(count == 1);
  assert(sizeof_int == sizeof(int));

  count = fread(&n, sizeof(int), 1, fid);
  assert(count == 1);

  v = ivector_create(n);

  count = fread(v->v, sizeof(int), n, fid);
  assert(count == n);

  fclose(fid);

  return v;
}

void ivector_set0(ivector *v) {
  assert(v);

  memset(v->v, 0, v->n * sizeof(int));
}

void ivector_export(const char *filename, const ivector *v) {
  FILE *fid;
  int n;
  int sizeof_int;

  assert(filename);
  assert(v);

  fid = fopen(filename, "w");
  assert(fid);

  sizeof_int = sizeof(int);
  n = fwrite(&sizeof_int, sizeof(int), 1, fid);
  assert(n == 1);

  n = fwrite(&(v->n), sizeof(int), 1, fid);
  assert(n == 1);

  n = fwrite(v->v, sizeof(int), v->n, fid);
  assert(n == v->n);

  fclose(fid);
}


/******************************************************************************/

/**********
 * svector
 **********/

svector *svector_create(int n, int nnz) {
  svector *v;

  assert(nnz > 0);
  assert(n > 0);
  assert(nnz <= n);

  v = malloc(sizeof(svector));
  assert(v);

  v->n = n;
  v->nnz = nnz;

  v->v = malloc(sizeof(elem) * nnz);
  assert(v->v);

  v->i = malloc(sizeof(int) * nnz);
  assert(v->i);

  return v;
}

svector *svector_import(const char *filename) {
  int n;
  size_t nnz;
  svector *v;
  FILE *fid;
  unsigned int r;
  int sizeof_elem;

  assert(filename);
  fid = fopen(filename, "r");
  assert(fid);

  r = fread(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);
  assert(sizeof_elem == sizeof(elem));

  r = fread(&n, sizeof(int), 1, fid);
  assert(r == 1);

  r = fread(&nnz, sizeof(int), 1, fid);
  assert(r == 1);

  v = svector_create(n, nnz);

  r = fread(v->v, sizeof(elem), nnz, fid);
  assert(r == nnz);

  r = fread(v->i, sizeof(int), nnz, fid);
  assert(r == nnz);

  fclose(fid);

  return v;
}

void svector_export(const char *filename, const svector *v) {
  FILE *fid;
  int r;
  int sizeof_elem;

  assert(filename);
  assert(v);

  fid = fopen(filename, "w");
  assert(fid);

  sizeof_elem = sizeof(elem);
  r = fwrite(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&(v->n), sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&(v->nnz), sizeof(int), 1, fid);
  assert(r == 1);


  r = fwrite(v->v, sizeof(elem), v->nnz, fid);
  assert(r == v->nnz);

  r = fwrite(v->i, sizeof(int), v->nnz, fid);
  assert(r == v->nnz);

  fclose(fid);
}

void svector_destroy(svector **v) {
  assert(v);
  assert(*v);

  free((*v)->v);
  free((*v)->i);
  free(*v);

  *v = NULL;
}

void svector_printf(const svector *v) {
  int i, index;

  assert(v);

  index = 0;
  for (i = 0; i < v->n; i++) {
    if (i == v->i[index]) {
      printf_elem_s(v->v[index]);
      index++;
    }
    else {
      printf_elem_s((elem) 0);
    }
  }

  assert(index == v->nnz);
}
