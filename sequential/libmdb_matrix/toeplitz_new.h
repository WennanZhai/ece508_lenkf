#ifndef TOEPLITZ_NEW_H
#define TOEPLITZ_NEW_H


#include "elem.h"
#include "util.h"
#include "vector.h"
#include "counter.h"
#include "sparse_coo.h"


/******************************************************************************/
/* sb_toe - symmetric Toeplitz matrix */

typedef struct {
  int n;
  elem *v_row0;
  int nnz_row0;
} s_toe;


s_toe *s_toe_create(int n, int nnz_row0);
void s_toe_destroy(s_toe **A);
void s_toe_printf(const s_toe *A);
s_toe *s_toe_import(const char *filename);
s_toe *s_toe_import_fid(FILE *fid);
void s_toe_export(const char *filename, const s_toe *A);
void s_toe_export_fid(FILE *fid, const s_toe *A);
elem s_toe_get(const s_toe *A, int i, int j);

typedef struct {
  const s_toe *A;
  int i;
  int j;
  elem v;
} s_toe_it;


void s_toe_iterator(const s_toe *A, s_toe_it *it, int i);
boolean s_toe_nz_it_has_next(const s_toe_it *it);
boolean s_toe_nz_next_check(const s_toe *A, int i, int j);
elem s_toe_nz_it_next(s_toe_it *it);
boolean s_toe_next_check(const s_toe *A, int j);
boolean s_toe_it_has_next(const s_toe_it *it);
elem s_toe_it_next(s_toe_it *it);



/******************************************************************************/
/* sb_toe - symmetric block Toeplitz matrix with symmetric Toeplitz blocks */

typedef struct {
  int n;	     /* A is n x n (= k*n_block x k*n_block) */
  int n_block;       /* Each block is n_block x n_block */
  int k;             /* The number of blocks */
  int k_nnz;         /* Number of non-zero blocks */
		     
  s_toe **A_block;   /* Array of pointers to each block */
} sb_toe;


sb_toe *sb_toe_create(int n_block, int k, int k_nnz, vector *nnz_row0_list);
void sb_toe_destroy(sb_toe **A);
void sb_toe_printf(const sb_toe *A);
sb_toe *sb_toe_import(const char *filename);
sb_toe *sb_toe_import_fid(FILE *fid);
void sb_toe_export(const char *filename, const sb_toe *A);
void sb_toe_export_fid(FILE *fid, const sb_toe *A);


typedef struct {
  const sb_toe *A;
  s_toe_it block_it;
  int i;
  int j;
  int block_i;
  int block_j;
  int block;
  elem v;
} sb_toe_it;

void sb_toe_iterator(const sb_toe *A, sb_toe_it *it, int i);
boolean sb_toe_nz_next_check(const sb_toe *A, int i, int j,
			     const s_toe_it *it);
boolean sb_toe_nz_it_has_next(const sb_toe_it *it);
elem sb_toe_nz_it_next(sb_toe_it *it);
boolean sb_toe_next_check(const sb_toe *A, int j,
			  const s_toe_it *block_it);
boolean sb_toe_it_has_next(const sb_toe_it *it);
elem sb_toe_it_next(sb_toe_it *it);



/******************************************************************************/
/* sb_toe_r - Recursive, symmetric, block toeplitz matrix */

typedef struct {
  int rank;
  int *n_phy;
  int *n;
  int *N_phy;
  int *N;
} sb_toe_r_dim;

typedef struct sb_toe_r_struct {
  sb_toe_r_dim *dim;

  union {
    elem *v;
    struct sb_toe_r_struct **A_block;
  } ptr;
} sb_toe_r;


sb_toe_r *sb_toe_r_create(int rank, int *n_phy, int *n);
void sb_toe_r_destroy(sb_toe_r **A);
sb_toe_r *sb_toe_r_import(const char *filename);
void sb_toe_r_export(const char *filename, const sb_toe_r *A);
void sb_toe_r_printf(const sb_toe_r *A);
void sb_toe_r_fprintf(FILE *fid, const sb_toe_r *A);
int sb_toe_r_nnz(const sb_toe_r *A);
sparse_coo *sb_toe_r_convert_coo(const sb_toe_r *A);


typedef struct sb_toe_r_it_struct {
  const sb_toe_r *A;
  int i;
  int j;
  int j_last;
  counter *j_state;
  int *i_state;
} sb_toe_r_it;


sb_toe_r_it *sb_toe_r_it_create(const sb_toe_r *A);
void sb_toe_r_it_init(sb_toe_r_it *A_it, int i);
void sb_toe_r_it_destroy(sb_toe_r_it **A_it);
boolean sb_toe_r_it_has_next(const sb_toe_r_it *A_it);
elem sb_toe_r_it_next(sb_toe_r_it *A_it);

sb_toe_r_it *sb_toe_r_nz_it_create(const sb_toe_r *A);
void sb_toe_r_nz_it_init(sb_toe_r_it *A_it, int i);
void sb_toe_r_nz_it_destroy(sb_toe_r_it **A_it);
boolean sb_toe_r_nz_it_has_next(const sb_toe_r_it *A_it);
elem sb_toe_r_nz_it_next(sb_toe_r_it *A_it);

#endif
