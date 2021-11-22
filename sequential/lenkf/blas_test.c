#ifdef LENKF_FLOAT_ELEM
#include "mdb_matrix_s.h"
#elif defined LENKF_DOUBLE_ELEM
#include "mdb_matrix_d.h"
#else
#error ?
#endif

#include "randn.h"


int main(void) {
  const int test_trans = 0;
  const int test_gemv = 1;
    
  randn_init();
  
  if (test_trans) {
    full_r *X;
    full_r *X_T;
    const int N = 256;
    const int L = 128;
    const int T = 2*64;
    const int M = 128;
    int t;
    vector *n;
    int i;
    elem alpha;
    stop_watch *sw[2];
    float t0, t1;
     
    X = full_r_create(N, L);
    X_T = full_r_create(L, N);
    
    full_r_set0(X);
    full_r_set0(X_T);
    
    for (i = 0; i < 2; i++) {
      sw[i] = stop_watch_create();
    }
    
    n = vector_create(N);
    
    for (t = 0; t < T*M; t++) {
      for (i = 0; i < N; i++) {
        n->v[i] = randn(1);
      }
      alpha = randn(1);
      
      for (i = 0; i < L; i++) {
        stop_watch_start(sw[0]);
        eaxpy(n->n, alpha, n->v, 1, &(X->v[0][i]), L);
        stop_watch_stop(sw[0]);
        
        stop_watch_start(sw[1]);
        eaxpy(n->n, alpha, n->v, 1, &(X->v[i][0]), 1);
        stop_watch_stop(sw[1]);
      }
    }
    
    full_r_destroy(&X);
    full_r_destroy(&X_T);
    
    t0 = sw[0]->sec + sw[0]->usec/1e6;
    t1 = sw[1]->sec + sw[1]->usec/1e6;
  
    for (i = 0; i < 2; i++) {
      printf("[%d] ", i);
      stop_watch_printf(sw[i]);
      stop_watch_destroy(&sw[i]);
    }
    printf("%f\n", t0 / t1);
    
    vector_destroy(&n);
  }

  if (test_gemv) {
    full_r *A;
    vector *b, *y;
    vector *b2, *y2;
    const int M = 4;
    const int N = 3;
    int i, j, count;
    
    A = full_r_create(M, N);
    b = vector_create(N);
    y = vector_create(M);
    b2 = vector_create(M);
    y2 = vector_create(N);
    
    count = 1;
    for (i = 0; i < M; i++) {
      for (j = 0; j < N; j++) {
        A->v[i][j] = count;
        count++;
      }
    }

    for (i = 0; i < N; i++) {
      b->v[i] = i+1;
    }

    for (i = 0; i < M; i++) {
      b2->v[i] = i+1;
    }
      
    egemv(CblasRowMajor, CblasNoTrans, M, N, 1,
          A->v_vector, N, b->v, 1, 0, y->v, 1);

    printf("y:\n");
    vector_printf(y);
    printf("\n");

    egemv(CblasRowMajor, CblasTrans, M, N, 1,
          A->v_vector, N, b2->v, 1, 0, y2->v, 1);

    printf("y2:\n");
    vector_printf(y2);
    
    full_r_destroy(&A);
    vector_destroy(&b);
    vector_destroy(&y);
    vector_destroy(&b2);
    vector_destroy(&y2);
  }
  
  randn_exit();
  
  return 0;
}
