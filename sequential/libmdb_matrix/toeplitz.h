#ifndef TOEPLITZ_H
#define TOEPLITZ_H

#include <stdio.h>

#include "sparse_rcs.h"
#include "elem.h"


/*************
 * s_toeplitz
 *************/

/* A symmetric Toeplitz matrix. */

typedef struct {
  int n;

  /* Sparse representation of the first row of the matrix.  The
     remainder of the matrix can be reconstructed from the first
     row. */
  elem *v_row0;
  int *j_row0;
  int nnz_row0;

  /* v and j are allocated with enough elements to represent the row
     with the most nonzero elements. */
  int row;
  elem *v;
  int *j;
  int nnz;
} s_toeplitz;

s_toeplitz *s_toeplitz_create(int n, int nnz_row0);
void s_toeplitz_destroy(s_toeplitz **A);
s_toeplitz *s_toeplitz_import(const char *filename);
s_toeplitz *s_toeplitz_import_fid(FILE *fid);
void s_toeplitz_export(const char *filename, const s_toeplitz *A);
void s_toeplitz_export_fid(FILE *fid, const s_toeplitz *A);
void s_toeplitz_get_row(s_toeplitz *A, int row);
void s_toeplitz_printf_row(s_toeplitz *A, int row);
void s_toeplitz_printf_current_row(s_toeplitz *A);
void s_toeplitz_printf(s_toeplitz *A);
int s_toeplitz_get_nnz(s_toeplitz *A);



/*******************
 * s_block_toeplitz
 *******************/

/* A symmetric block Toeplitz matrix with symmetric Toeplitz blocks. */
/* Each block is the same size. */

typedef struct {
  int n;            /* A is n x n  (= K*n_block x K*n_block) */


  s_toeplitz **B;   /* An array of pointers to the blocks */
  int K;            /* The number of blocks */
  int n_block;      /* Each block is n_block x n_block */

  /* v and j are allocated with enough elements to represent the row
     with the most nonzero elements. */
  int row;
  elem *v;
  int *j;
  int nnz;
  int max_row_length;
} s_block_toeplitz;

s_block_toeplitz *s_block_toeplitz_create(int K);
void s_block_toeplitz_destroy(s_block_toeplitz **A);
s_block_toeplitz *s_block_toeplitz_import(const char *filename);
void s_block_toeplitz_export(const char *filename, const s_block_toeplitz *A);
void s_block_toeplitz_printf_blocks(const s_block_toeplitz *A);
void s_block_toeplitz_get_row(s_block_toeplitz *A, int row);
void s_block_toeplitz_printf_row(s_block_toeplitz *A, int row);
elem s_block_toeplitz_get(s_block_toeplitz *A, int i, int j);
int s_block_toeplitz_get_nnz(s_block_toeplitz *A);
sparse_rcs *s_block_toeplitz_convert(s_block_toeplitz *A);

#endif
