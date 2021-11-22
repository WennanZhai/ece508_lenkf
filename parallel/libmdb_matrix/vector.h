#ifndef VECTOR_H
#define VECTOR_H

#include <stdio.h>

#include "elem.h"


/*********
 * vector
 *********/

/* Vector of doubles or floats. */

typedef struct {
  elem *v;
  int n;
} vector;


vector *vector_create(int n);
vector *vector_import(const char *filename);
void vector_export(const char *filename, const vector *v);
void vector_destroy(vector **v);
void vector_printf(const vector *v);
void vector_fprintf(FILE *fid, const vector *v);
void vector_fnprintf(FILE *fid, const vector *v, int n);
void vector_set0(vector *v);
void vector_set_nan(vector *v);
void vector_copy(vector *y, const vector *x);
elem vector_norm(const vector *x);
elem vector_squared_norm(const vector *x);
void vector_add_scalar(vector *x, elem alpha);
void vector_scal(vector *x, elem alpha);


/**********
 * zvector
 **********/

/* Vector of complex numbers. */

typedef struct {
  z_elem *v;
  int n;
} zvector;

zvector *zvector_create(int n);
void zvector_destroy(zvector **v);
void zvector_export(const char *filename, const zvector *v);
void zvector_printf(const zvector *v);
void zvector_set0(const zvector *v);



/**********
 * ivector
 **********/

/* Vector of integers. */

typedef struct {
  int *v;
  int n;
} ivector;


ivector *ivector_create(int n);
ivector *ivector_import(const char *filename);
void ivector_export(const char *filename, const ivector *v);
void ivector_destroy(ivector **v);
void ivector_printf(const ivector *v);
void ivector_set0(ivector *v);


/**********
 * svector
 **********/

/* Sparse vector of doubles or floats. */

typedef struct {
  elem *v;
  int *i;
  int nnz, n;
} svector;


svector *svector_create(int n, int nz);
svector *svector_import(const char *filename);
void svector_export(const char *filename, const svector *v);
void svector_destroy(svector **v);
void svector_printf(const svector *v);




#endif
