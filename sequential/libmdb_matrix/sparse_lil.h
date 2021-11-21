#ifndef SPARSE_LIL_H
#define SPARSE_LIL_H

#include "elem.h"
#include "llist.h"
#include "sparse_rcs.h"
#include "full_c.h"


extern const int SPARSE_LIL_BLOCK_SIZE;


typedef struct {
  elem *v;
  int  *j;
  int count;
} sparse_lil_node;
  
typedef struct sparse_lil_struct {
  llist **row;
  int m, n, N;
  int block_size;
} sparse_lil;


sparse_lil *sparse_lil_create(int m, int n);
sparse_lil *sparse_lil_create_block_size(int m, int n, int block_size);
void sparse_lil_destroy(sparse_lil **A);
void sparse_lil_append(sparse_lil *A, int i, int j, elem v);
void sparse_lil_printf(const sparse_lil *A);
void sparse_lil_mmm(sparse_rcs *A, sparse_lil *B, full_c *C);
void sparse_lil_foreach(sparse_lil *A, void (*node_fun)(sparse_lil_node *));
void sparse_lil_scal(sparse_lil *A, elem alpha);
int sparse_lil_row_max(sparse_lil *A);
sparse_rcs *sparse_lil_2_rcs(sparse_lil *A);


typedef struct {
  int i;
  int j;
  llist_it row_it;
  int node_idx;
  sparse_lil *A;
} sparse_lil_row_it;

void sparse_lil_row_it_init(sparse_lil *A, int i, sparse_lil_row_it *it);
boolean sparse_lil_row_it_has_next(sparse_lil_row_it *it);
elem sparse_lil_row_it_next(sparse_lil_row_it *it);


#endif
