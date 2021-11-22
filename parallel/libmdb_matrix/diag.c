#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "diag.h"
#include "vector.h"
#include "util.h"


diag *diag_create(int m, int n) {
  diag *d;

  assert(m > 0);
  assert(n > 0);

  d = malloc(sizeof(diag));
  assert(d);

  d->m = m;
  d->n = n;

  d->v = vector_create(MIN(m, n));
  vector_set0(d->v);
  
  return d;
}

void diag_destroy(diag **d) {
  assert(*d);

  vector_destroy(&(*d)->v);
  free(*d);

  *d = NULL;
}


diag *diag_import(const char *filename) {
  diag *d;
  FILE *fid;
  int c, m, n, sizeof_elem;
  
  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  c = fread(&sizeof_elem, sizeof(int), 1, fid);
  assert(c == 1);
  assert(sizeof_elem == sizeof(elem));

  c = fread(&m, sizeof(int), 1, fid);
  c += fread(&n, sizeof(int), 1, fid);
  assert(c == 2);
  
  d = diag_create(m, n);

  c = fread(d->v->v, sizeof(elem), MIN(m, n), fid);
  assert(c == MIN(m, n));

  fclose(fid);
  
  return d;
}


void diag_export(const char *filename, const diag *d) {
  FILE *fid;
  int c;
  int sizeof_elem;
  
  assert(filename);
  assert(d);

  fid = fopen(filename, "w");
  assert(fid);

  sizeof_elem = sizeof(elem);
  c = fwrite(&sizeof_elem, sizeof(int), 1, fid);
  assert(c == 1);

  c = fwrite(&d->m, sizeof(int), 1, fid);
  c += fwrite(&d->n, sizeof(int), 1, fid);
  assert(c == 2);
  
  c = fwrite(d->v->v, sizeof(elem), MIN(d->m, d->n), fid);
  assert(c == MIN(d->m, d->n));
  
  fclose(fid);
}


elem diag_get(const diag *d, int i) {
  assert(d);
  assert(i < MIN(d->m, d->n));

  return d->v->v[i];
}


void diag_set(const diag *d, int i, elem e) {
  assert(d);
  assert(i < MIN(d->m, d->n));

  d->v->v[i] = e;
}


void diag_printf(const diag *d) {
  vector_printf(d->v);
}


void diag_scal(const diag *d, const elem alpha) {
  assert(d);

  vector_scal(d->v, alpha);
}
