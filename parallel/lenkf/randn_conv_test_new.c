#ifdef LENKF_FLOAT_ELEM
#include "mdb_matrix_s.h"
#elif defined LENKF_DOUBLE_ELEM
#include "mdb_matrix_d.h"
#else
#error ?
#endif


#include <stdlib.h>
#include <assert.h>

#include "ensemble.h"
#include "randn.h"


extern boolean RANDN_DEBUG;


int main(int argc, char **argv) {
  FILE *fid;
    
  int L;
  int N;
  int *n;
  int rank;
  
  int l, i, r;

  r_filter_new *h;
  r_filter_new *u;
  
  ensemble *e;


  RANDN_DEBUG = True;
  
  assert(argc == 4);

  fid = fopen(argv[1], "r");
  assert(fid);

  r = fread(&rank, sizeof(int), 1, fid);
  assert(r == 1);

  assert(rank > 0);
  n = malloc(rank * sizeof(int));
  assert(n);

  r = fread(n, sizeof(int), rank, fid);
  assert(r == rank);

  r = fread(&L, sizeof(int), 1, fid);
  assert(r == 1);
  assert(L > 0);
  fclose(fid);


  randn_init();

  h = r_filter_new_import(argv[2]);
  r_filter_new_dft(h);

  u = r_filter_new_create_same_dim(h);

  rank = h->rank;
  
  N = 1;
  for (i = 0; i < rank; i++) {
    N *= n[i];
  }

  e = ensemble_create(N, L);
  ensemble_set0(e);

  for (l = 0; l < L; l++) {
    randn_v_conv_add(&e->X->v[0][l], L, u, h, rank, n);
  }

  /*
  printf("e:\n");
  ensemble_printf(e);
  */
  
  full_r_export(argv[3], e->X);
  
  r_filter_new_destroy(&h);
  r_filter_new_destroy(&u);

  ensemble_destroy(&e);

  free(n);
  
  randn_exit();
  
  fftwe_cleanup();
  
  return 0;
}
