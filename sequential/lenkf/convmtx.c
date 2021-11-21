#ifdef LENKF_FLOAT_ELEM
#include "mdb_matrix_s.h"
#elif defined LENKF_DOUBLE_ELEM
#include "mdb_matrix_d.h"
#else
#error ?
#endif


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


#define ZERO 1e-6

static void r_filter_new_2_sparse_lil(int i, const r_filter_new *X,
				      sparse_lil *C_lil);


static void r_filter_new_2_sparse_lil(int i, const r_filter_new *X,
				      sparse_lil *C_lil) {
  int j;
  elem *e_ptr;
  int rank;
  int index;
  
  assert(i >= 0);
  assert(i < C_lil->m);
  assert(X);
  assert(X->d == S);
  assert(C_lil);

  rank = X->rank;

  //printf("N_log[0] = %d\n", X->N_log[0]);
  //printf("N_phy[0] = %d\n", X->N_phy[0]);

  //printf("N_phy = [");
  //for (j = 0; j < rank - 1; j++) {
  //printf("%d, ",  X->N_phy[j]);
  //}
  //printf("%d]\n", X->N_phy[rank - 1]);
  
  e_ptr = X->h;
  index = 0;
  for (j = 0; j < X->N_log[0]; j++) {
    //printf("i=%d j=%d v(%d)=", i, j, index);
    //printf_elem_n(*e_ptr);
    
    if (fabse(*e_ptr) > ZERO) {
      sparse_lil_append(C_lil, i, j, *e_ptr);
    }

    if (j % X->n_log[rank - 1] == X->n_log[rank - 1] - 1) {
      //printf("--> %d %d\n", X->N_phy[rank - 1], X->N_log[rank - 1]);
      
      e_ptr += X->N_phy[rank - 1] - X->N_log[rank - 1] + 1;
      index += X->N_phy[rank - 1] - X->N_log[rank - 1] + 1;
    }
    else {
      e_ptr++;
      index++;
    }
  }
}


int main(int argc, char **argv) {
  sparse_lil *C_T_lil;
  sparse_rcs *C_T_rcs;
  r_filter_new *H;
  r_filter_new *X;
  int rank;
  int *n;
  const int *m;
  int M, N;
  FILE *fid;
  int c;
  int i;
  elem *e_ptr;
  counter *state;
  
  H = r_filter_new_import("/tmp/H_convmtx_test");
  r_filter_new_dft(H);

  X = r_filter_new_create_same_dim(H);
  
  fid = fopen("/tmp/n_convmtx_test", "r");
  assert(fid);

  c = fread(&rank, sizeof(int), 1, fid);
  assert(c == 1);

  assert(rank > 0);
  n = malloc(rank * sizeof(int));
  assert(n);

  c = fread(n, sizeof(int), rank, fid);
  assert(c == rank);

  fclose(fid);

  m = H->n_log;
  //m = malloc(rank * sizeof(int));
  //assert(m);

  M = 1;
  N = 1;
  for (i = 0; i < rank; i++) {
    //m[i] = H->n_log[i] + n[i] - 1;
    M *= m[i];
    N *= n[i];
  }

  //C_lil = sparse_lil_create(M, N);
  C_T_lil = sparse_lil_create(N, M);

  state = counter_create(rank, X->n_log);
  counter_reset(state);
  
  //for (i = 0; i < M; i++) {
  for (i = 0; i < N; i++) {
    r_filter_new_set0(X);
    e_ptr = r_filter_new_get_ptr(X, state->c);
    *e_ptr = 1;

    r_filter_new_dft(X);
    //printf("\ndft(X):\n");
    //r_filter_new_printf_dft(X);

    //printf("\ndft(H):\n");
    //r_filter_new_printf_dft(H);
    
    r_filter_new_execute_r(H, X);
    r_filter_new_idft(X);

    //printf("\nidft(dft(X).*dft(H)):\n");
    //r_filter_new_printf(X);
	   
    r_filter_new_2_sparse_lil(i, X, C_T_lil);

    /*
    counter_printf(state);
    printf("\n");
    counter_max_printf(state);
    printf("\n");
    printf("i=%d M=%d N=%d\n", i, M, N);
    fflush(stdout);
    */
    
    counter_tick(state);
  }

  C_T_rcs = sparse_lil_2_rcs(C_T_lil);

  sparse_rcs_export("/tmp/C_T_convmtx_test", C_T_rcs);
  
  free(n);
  //free(m);
  
  r_filter_new_destroy(&H);
  r_filter_new_destroy(&X);
  sparse_lil_destroy(&C_T_lil);
  sparse_rcs_destroy(&C_T_rcs);
  counter_destroy_keep_max(&state);

  fftwe_cleanup();
  
  return 0;
}

