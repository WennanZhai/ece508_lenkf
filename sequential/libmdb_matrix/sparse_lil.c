#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "sparse_lil.h"
#include "eblas.h"


const int SPARSE_LIL_BLOCK_SIZE = 4;


static sparse_lil_node *create_node(const int block_size);
static void destroy_node(void *node_ptr_void);
static void printf_node(void *node_ptr_void);

static void node_fun_wrap(void *node_ptr_void);
static void (*NODE_FUN_WRAP_F)(sparse_lil_node *);

static void scale_node(sparse_lil_node *node_ptr);
static elem SCALE_NODE_ALPHA;

static void copy_node(void *node_ptr_void);
static int COPY_NODE_IDX;
typedef struct v_j_pair {
  elem v;
  int j;
} vj_pair;
static vj_pair *COPY_NODE_LOC;
static int vj_pair_compare(const void *a, const void *b);

static void get_row_count(void *node_ptr_void);
static int ROW_COUNT;


sparse_lil *sparse_lil_create(int m, int n) {
  return sparse_lil_create_block_size(m, n, SPARSE_LIL_BLOCK_SIZE);
}


sparse_lil *sparse_lil_create_block_size(int m, int n, int block_size) {
  sparse_lil *A;

  assert(m > 0);
  assert(n > 0);
  assert(block_size > 0);

  A = malloc(sizeof(sparse_lil));
  assert(A);

  A->row = malloc(m * sizeof(llist *));
  assert(A->row);

  /*/
#ifdef LONG_PTR
  memset((void *) A->row, (long int) NULL, m * sizeof(llist *));
#else
  memset((void *) A->row, (int) NULL, m * sizeof(llist *));
#endif
  */

  memset((void *) A->row, (uintptr_t) NULL, m * sizeof(llist *));

  A->m = m;
  A->n = n;
  A->N = 0;

  A->block_size = block_size;

  return A;
}


static void destroy_node(void *node_ptr_void) {
  sparse_lil_node *node_ptr;

  assert(node_ptr_void);

  node_ptr = (sparse_lil_node *) node_ptr_void;

  free(node_ptr->v);
  free(node_ptr->j);
  free(node_ptr);
}


void sparse_lil_destroy(sparse_lil **A) {
  int i;

  assert(A);
  assert(*A);

  for (i = 0; i < (*A)->m; i++) {
    if ((*A)->row[i]) {
      llist_destroy(&(*A)->row[i], &destroy_node);
    }
  }

  free((*A)->row);
  free(*A);

  *A = NULL;
}


static sparse_lil_node *create_node(const int block_size) {
  sparse_lil_node *node_ptr;

  node_ptr = malloc(sizeof(sparse_lil_node));
  assert(node_ptr);

  node_ptr->v = malloc(block_size * sizeof(elem));
  assert(node_ptr->v);

  node_ptr->j = malloc(block_size * sizeof(int));
  assert(node_ptr->j);

  node_ptr->count = 0;

  return node_ptr;
}


void sparse_lil_append(sparse_lil *A, int i, int j, elem v) {
  sparse_lil_node *node_ptr;

  assert(A);
  assert(i >= 0);
  assert(i < A->m);
  assert(j >= 0);
  assert(j < A->n);

  if (A->row[i] == NULL) {
    A->row[i] = llist_create();

    node_ptr = create_node(A->block_size);
    node_ptr->count = 1;
    node_ptr->v[0] = v;
    node_ptr->j[0] = j;

    llist_append(A->row[i], node_ptr);

    A->N++;

    return;
  }

  node_ptr = (sparse_lil_node *) A->row[i]->tail->data;

  if (node_ptr->count == A->block_size) {
    node_ptr = create_node(A->block_size);
    llist_append(A->row[i], node_ptr);
  }

  node_ptr->v[node_ptr->count] = v;
  node_ptr->j[node_ptr->count] = j;
  node_ptr->count++;

  A->N++;
}


static void printf_node(void *node_ptr_void) {
  int i;

  sparse_lil_node *node_ptr;

  assert(node_ptr_void);

  node_ptr = (sparse_lil_node *) node_ptr_void;

  for (i = 0; i < node_ptr->count; i++) {
    printf("(");
    printf_elem(node_ptr->v[i]);
    printf(", %d) ", node_ptr->j[i]);
  }
}


void sparse_lil_printf(const sparse_lil *A) {
  int i;

  assert(A);

  for (i = 0; i < A->m; i++) {
    if (A->row[i]) {
      llist_foreach(A->row[i], &printf_node);
      printf("\n");
    }
    else {
      printf("<empty>\n");
    }
  }
}


/* C = C + A*B */
void sparse_lil_mmm(sparse_rcs *A, sparse_lil *B, full_c *C) {
  int i, j;
  int index;
  sparse_lil_row_it B_it;
  elem A_v, B_v;

  assert(A);
  assert(B);
  assert(C);

  assert(A->m == C->m);
  assert(B->n == C->n);
  assert(A->n == B->m);

  for (i = 0; i < A->m; i++) {
    for (index = A->r[i]; index < A->r[i+1]; index++) {
      A_v = A->v[index];
      j = A->j[index];

      sparse_lil_row_it_init(B, j, &B_it);

      while (sparse_lil_row_it_has_next(&B_it)) {
	B_v = sparse_lil_row_it_next(&B_it);

	/* Matrix indexed in [j][i] because it is stored column-major */
	C->v[B_it.j][i] += A_v * B_v;
      }
    }
  }
}


static void node_fun_wrap(void *node_ptr_void) {
  sparse_lil_node *node;

  assert(node_ptr_void);

  node = (sparse_lil_node *) node_ptr_void;
  NODE_FUN_WRAP_F(node);
}


void sparse_lil_foreach(sparse_lil *A,
			void (*node_fun)(sparse_lil_node *)) {
  int i;

  assert(A);
  assert(node_fun);

  NODE_FUN_WRAP_F = node_fun;

  for (i = 0; i < A->m; i++) {
    if (A->row[i]) {
      llist_foreach(A->row[i], &node_fun_wrap);
    }
  }
}


static void scale_node(sparse_lil_node *node_ptr) {
  escal(node_ptr->count, SCALE_NODE_ALPHA, node_ptr->v, 1);
}


void sparse_lil_scal(sparse_lil *A, elem alpha) {
  assert(A);

  SCALE_NODE_ALPHA = alpha;

  sparse_lil_foreach(A, &scale_node);
}


static void get_row_count(void *node_ptr_void) {
  sparse_lil_node *node_ptr;

  assert(node_ptr_void);

  node_ptr = (sparse_lil_node *) node_ptr_void;

  ROW_COUNT += node_ptr->count;
}


int sparse_lil_row_max(sparse_lil *A) {
  int i;
  int row_max;

  assert(A);

  row_max = 0;
  for (i = 0; i < A->m; i++) {
    if (A->row[i]) {
      ROW_COUNT = 0;
      llist_foreach(A->row[i], &get_row_count);
      if (ROW_COUNT > row_max) {
	row_max = ROW_COUNT;
      }
    }
  }

  return row_max;
}


static void copy_node(void *node_ptr_void) {
  sparse_lil_node *node_ptr;
  int i;

  assert(node_ptr_void);
  assert(COPY_NODE_LOC);
  assert(COPY_NODE_IDX >= 0);

  node_ptr = (sparse_lil_node *) node_ptr_void;

  for (i = 0; i < node_ptr->count; i++) {
    COPY_NODE_LOC[COPY_NODE_IDX].v = node_ptr->v[i];
    COPY_NODE_LOC[COPY_NODE_IDX].j = node_ptr->j[i];
    COPY_NODE_IDX++;
  }
}


static int vj_pair_compare(const void *a, const void *b) {
  const sparse_lil_node *a_node;
  const sparse_lil_node *b_node;

  assert(a);
  assert(b);

  a_node = (sparse_lil_node *) a;
  b_node = (sparse_lil_node *) b;

  return a_node->j > b_node->j;
}


/* A very simple implemntation of sparse_lil to sparse_rcs conversion */
sparse_rcs *sparse_lil_2_rcs(sparse_lil *A) {
  sparse_rcs *B;
  vj_pair *sorted_node;
  int i, k, index;;
  int row_max;

  B = sparse_rcs_create(A->m, A->n, A->N);

  row_max = sparse_lil_row_max(A);
  sorted_node = malloc(row_max * sizeof(vj_pair));
  assert(sorted_node);

  B->r[0] = 0;

  index = 0;
  for (i = 0; i < A->m; i++) {
    COPY_NODE_IDX = 0;
    COPY_NODE_LOC = sorted_node;

    if (A->row[i]) {
      llist_foreach(A->row[i], &copy_node);
      qsort(sorted_node, COPY_NODE_IDX, sizeof(vj_pair), &vj_pair_compare);

      for (k = 0; k < COPY_NODE_IDX; k++) {
	B->v[index] = sorted_node[k].v;
	B->j[index] = sorted_node[k].j;
	index++;
      }
    }

    B->r[i+1] = COPY_NODE_IDX;
  }

  for (i = 2; i < A->m + 1; i++) {
    B->r[i] = B->r[i-1] + B->r[i];
  }

  free(sorted_node);

  return B;
}


void sparse_lil_row_it_init(sparse_lil *A, int i, sparse_lil_row_it *it) {
  assert(A);
  assert(i >= 0);
  assert(i < A->m);
  assert(it);

  it->A = A;
  it->i = i;

  if (A->row[i]) {
    llist_it_init(&it->row_it, A->row[i]);
  }
  else {
    it->row_it.l = NULL;
    it->row_it.n = NULL;
  }

  it->node_idx = -1;
  it->j = -1;
}


boolean sparse_lil_row_it_has_next(sparse_lil_row_it *it) {
  sparse_lil_node *node;

  assert(it);

  if (it->row_it.l == NULL) {
    assert(it->row_it.n == NULL);
    return False;
  }
  else if (llist_it_has_next(&it->row_it)) {
    return True;
  }
  else {
    if (it->row_it.n) {
      node = (sparse_lil_node *) it->row_it.n->data;
      return it->node_idx < node->count;
    }
    else {
      return False;
    }
  }

}


elem sparse_lil_row_it_next(sparse_lil_row_it *it) {
  sparse_lil_node *node;

  assert(it);
  assert(sparse_lil_row_it_has_next(it));

  if (it->node_idx == -1) {
    node = (sparse_lil_node *)  llist_it_next(&it->row_it);
    it->node_idx = 0;
  }
  else {
    node = (sparse_lil_node *) it->row_it.n->data;

    if (it->node_idx == node->count) {
      node = (sparse_lil_node *)  llist_it_next(&it->row_it);
      it->node_idx = 0;
    }
  }

  it->node_idx++;
  it->j = node->j[it->node_idx-1];
  return node->v[it->node_idx-1];
}
