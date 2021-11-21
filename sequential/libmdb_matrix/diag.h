#ifndef DIAG_H
#define DIAG_H

#include "vector.h"

typedef struct {
  int m, n;
  vector *v;
} diag;


diag *diag_create(int m, int n);
void diag_destroy(diag **d);
diag *diag_import(const char *filename);
void diag_export(const char *filename, const diag *d);
elem diag_get(const diag *d, int i);
void diag_set(const diag *d, int i, elem e);
void diag_printf(const diag *d);
void diag_scal(const diag *d, const elem alpha);

#endif
