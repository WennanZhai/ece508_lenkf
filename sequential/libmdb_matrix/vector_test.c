#ifdef FLOAT_ELEM_TEST
#include "mdb_matrix_s.h"
#elif defined DOUBLE_ELEM_TEST
#include "mdb_matrix_d.h"
#else
#error 0
#endif


int main(int argc, char **argv) {
  vector *a;
  int N = 4;
  int i;

  zvector *b;
  z_elem alpha;
  
  a = vector_create(N);
  b = zvector_create(N);
  
  for (i = 0; i < N; i++) {
    a->v[i] = i;

    b->v[i][0] = i;
    b->v[i][1] = i;
  }

  printf("a:\n");
  vector_printf(a);

  printf("\nb:\n");
  zvector_printf(b);

  alpha[0] = 0;
  alpha[1] = 0;
  
  zescal(b->n, alpha, b->v, 1);
  
  printf("\nalpha*b:\n");
  zvector_printf(b);

  vector_destroy(&a);
  zvector_destroy(&b);
  
  return 0;
}
