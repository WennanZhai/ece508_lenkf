#include <math.h>
#include <assert.h>

#include "randn.h"
#include "ensemble.h"


static void init_G(sparse_rcs *G);

void init_G(sparse_rcs *G) {
  assert(G);

  G->v = malloc(sizeof(elem) * 3);
  assert(G->v);

  G->j = malloc(sizeof(int) * 3);
  assert(G->j);

  G->N = 3;

  G->v[0] = 1.41421356237310;
  G->v[1] = 0.70710678118655;
  G->v[2] = 1.22474487139159;

  G->j[0] = 0;
  G->j[1] = 0;
  G->j[2] = 1;

  G->r[0] = 0;
  G->r[1] = 1;
  G->r[2] = 3;
}

int main(void) {
  sparse_rcs *G;
  ensemble *e;
  const int N = pow(2, 16);
  /*const int N = pow(2, 2);*/
  vector *mean;
  full_r *C;
  vector *scratch;
  int i;


  /* C = G' * G */
  /* G: Cholesky factorization of C */

  randn_init();

  printf("First 4 random numbers:\n");
  for (i = 0; i < 4; i++) {
    printf("%f\t", randn(1));
  }
  printf("\n");

  randn_reset();

  printf("\n");
  printf("First 4 random numbers (after reset):\n");
  for (i = 0; i < 4; i++) {
    printf("%f\t", randn(1));
  }
  printf("\n");


  G = sparse_rcs_create(2, 2, 0);
  init_G(G);

  /*G = sparse_rcs_import("/tmp/Q_sqrt");
    sparse_rcs_printf(G);*/
  /* Answer:
     1.414214        0.000000
     0.707107        1.224745  */

  /*
    G is the Cholesky factorization of:

    A = [2 1;
         1 2]

    i.e., A = G * G'

  */

  e = ensemble_create(G->m, N);
  ensemble_set0(e);
  scratch = vector_create(e->N);
  ensemble_add_noise(e, G, scratch);

  mean = vector_create(G->m);
  ensemble_mean(e, mean);
  printf("\n");
  printf("mean:\n");
  vector_printf(mean);
  /* Answer (in the limit):
     +0.000000 +0.000000 */

  C = full_r_create(G->m, G->m);
  ensemble_cov(e, C);
  printf("\n");
  printf("cov:\n");
  full_r_printf(C);
  /* Answer (in the limit):
     +2.000000 +1.000000
     +1.000000 +2.000000 */

  sparse_rcs_destroy(&G);
  ensemble_destroy(&e);
  vector_destroy(&mean);
  full_r_destroy(&C);
  vector_destroy(&scratch);

  randn_exit();

  return 0;
}
