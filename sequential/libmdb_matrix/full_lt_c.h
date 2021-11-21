#ifndef FULL_LT_C_H
#define FULL_LT_C_H

#include "elem.h"
#include "full_c.h"


typedef struct {
  int m, n;
  elem *v;
  int N;
} lt_c;


lt_c *lt_c_create(int m, int n);
void lt_c_destroy(lt_c **A);
lt_c *lt_c_import(const char *filename);
elem lt_c_get(const lt_c *A, const int i, const int j);
elem lt_c_get_submatrix(const lt_c *A, const int i, const int j,
			const int M);
void lt_c_set(const lt_c *A, const int i, const int j, const elem v);
void lt_c_add(const lt_c *A, const int i, const int j, const elem v);
void lt_c_printf(const lt_c *A);
void lt_c_fprintf(FILE *fid, const lt_c *A);
void lt_c_submatrix_fprintf(FILE *fid, const lt_c *A, const int M);
void lt_c_set0(lt_c *A);
void lt_c_eppsv(lt_c *A, full_c *B);

#endif
