#include <assert.h>
#include <math.h>

#ifdef FLOAT_ELEM_TEST
#include "mdb_matrix_s.h"
#elif defined DOUBLE_ELEM_TEST
#include "mdb_matrix_d.h"
#else
#error 0
#endif


int main(int argc, char **argv) {
  full_r *X;
  full_r *A;
  full_r *B;
  const int N = 6;
  const int K = 2;
  int i, j, index, r;

  full_c *X_c;
  full_c *Y_c;
  full_c *A_c;
  full_c *B_c;
  full_c *B_c2;

  ut_c *A_ut_c;

  const int N2 = 1000;
  const int K2 = 700;

  multi_sw *sw;


  X = full_r_create(N, N);
  A = full_r_create(N, N);
  B = full_r_create(N, K);

  for (i = 0; i < N*N; i++) {
    X->v_vector[i] = sin(cos(i));
  }

  printf("X:\n");
  full_r_printf(X);

  /* Result:

     X:
     +0.841471 +0.514395 -0.404239 -0.836022 -0.608083 +0.279873
     +0.819289 +0.684489 -0.144987 -0.790197 -0.744023 +0.004426
     +0.747210 +0.787934 +0.136312 -0.688695 -0.817847 -0.271704
     +0.613367 +0.835315 +0.396850 -0.520750 -0.841450 -0.507976
     +0.411573 +0.836685 +0.602731 -0.288001 -0.820683 -0.680216
     +0.153640 +0.792405 +0.740775 -0.013276 -0.750336 -0.785617

     Matches Matlab:

     >> X = reshape(sin(cos(0:35)),6,6)'

     X =

      0.8415    0.5144   -0.4042   -0.8360   -0.6081    0.2799
      0.8193    0.6845   -0.1450   -0.7902   -0.7440    0.0044
      0.7472    0.7879    0.1363   -0.6887   -0.8178   -0.2717
      0.6134    0.8353    0.3968   -0.5208   -0.8414   -0.5080
      0.4116    0.8367    0.6027   -0.2880   -0.8207   -0.6802
      0.1536    0.7924    0.7408   -0.0133   -0.7503   -0.7856

  */


  full_r_mmmT(X, X, 0, A);

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      if (i > j) {
	A->v[i][j] = 0;
      }
    }
  }

  printf("\n");
  printf("A:\n");
  full_r_printf(A);

  /* Result:

     A:
     +2.283112 +2.214404 +1.976003 +1.590251 +1.082510 +0.484936
     +0.000000 +2.338782 +2.283249 +2.052055 +1.657685 +1.126147
     +0.000000 +0.000000 +2.414742 +2.355416 +2.103299 +1.676400
     +0.000000 +0.000000 +0.000000 +2.468717 +2.376608 +2.087480
     +0.000000 +0.000000 +0.000000 +0.000000 +2.451879 +2.326717
     +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +2.380633

     Matches Matlab:

     >> A = X*X'

     A =

      2.2831    2.2144    1.9760    1.5903    1.0825    0.4849
      2.2144    2.3388    2.2832    2.0521    1.6577    1.1261
      1.9760    2.2832    2.4147    2.3554    2.1033    1.6764
      1.5903    2.0521    2.3554    2.4687    2.3766    2.0875
      1.0825    1.6577    2.1033    2.3766    2.4519    2.3267
      0.4849    1.1261    1.6764    2.0875    2.3267    2.3806

   */

#ifndef VECLIB
  r = epotrf(LAPACK_ROW_MAJOR, 'U', N, A->v_vector, A->n);
  assert(r == 0);

  printf("\n");
  printf("chol(A):\n");
  full_r_printf(A);

  /* Result (using double precision):

     chol(A):
     +1.510997 +1.465525 +1.307748 +1.052452 +0.716421 +0.320938
     +0.000000 +0.437057 +0.839049 +1.166121 +1.390555 +1.500502
     +0.000000 +0.000000 +0.023121 +0.027797 -0.014817 -0.099473
     +0.000000 +0.000000 +0.000000 +0.021248 +0.068965 +0.127390
     +0.000000 +0.000000 +0.000000 +0.000000 +0.000532 +0.001743
     +0.000000 +0.000000 +0.000000 +0.000000 +0.000000 +0.000619


     Matches Matlab:

     >> chol(A)

     ans =

     1.5110    1.4655    1.3077    1.0525    0.7164    0.3209
          0    0.4371    0.8390    1.1661    1.3906    1.5005
          0         0    0.0231    0.0278   -0.0148   -0.0995
	  0         0         0    0.0212    0.0690    0.1274
	  0         0         0         0    0.0005    0.0017
	  0         0         0         0         0    0.0006

   */

  for (i = 0; i < N*K; i++) {
    B->v_vector[i] = i + 1;
  }

  printf("\n");
  printf("B:\n");
  full_r_printf(B);

  /* Result:

     B:
     +1.000000 +2.000000
     +3.000000 +4.000000
     +5.000000 +6.000000
     +7.000000 +8.000000
     +9.000000 +10.000000
     +11.000000 +12.000000

     Matches Matlab:

     >> B = reshape(1:12,2,6)'

     B =

      1     2
      3     4
      5     6
      7     8
      9    10
     11    12

  */

  r = epotrs(LAPACK_ROW_MAJOR, 'U', N, K, A->v_vector,
	     A->n, B->v_vector, 2); //N);
  assert(r == 0);

  printf("\n");
  printf("N=%d\n", N);
  printf("A\\B:\n");
  full_r_printf(B);

  /* Result:

     A\B:
     +912940.465040 +1101470.369246
     -2016942.977022 -2493278.049158
     +1086199.119413 +1504851.031842
     +1321524.851915 +1321954.918207
     -2145319.349255 -2393575.574512
     +941202.796768 +1075576.998603


     NOTE: This is from the ATLAS FAQ:

     What's the deal with the RHS in the row-major
     factorization/solves?

     Most users are confused by the row major factorization and
     related solves. The right-hand side vectors are probably the
     biggest source of confusion. The RHS array does not represent a
     matrix in the mathematical sense, it is instead a pasting
     together of the various RHS into one array for calling
     convenience. As such, RHS vectors are always stored contiguously,
     regardless of the row/col major that is chosen. This means that
     ldb/ldx is always independent of NRHS, and dependant on N,
     regardless of the row/col major setting.

     In other words, ATLAS does not think of B as a matrix (in row or
     column major order).  Instead, it is a contiguous block of memory
     with each RHS vector stored in order, i.e., stacked on top of
     each other.  This is column major order in the mdb_matrix
     framework!


     Matches Matlab:

     >> A\B

     ans =

      1.0e+06 *

      0.912940448070594   1.101470350388620
     -2.016942918637024  -2.493277985310851
      1.086199031417526   1.504850937411045
      1.321524922281788   1.321954991701341
     -2.145319377152586  -2.393575602154475
      0.941202800133764   1.075577001309223

  */


  full_r_destroy(&X);
  full_r_destroy(&A);
  full_r_destroy(&B);

#endif

  /****************************************************************************/


  sw = multi_sw_create(2);
  multi_sw_set_name(sw, 0, "po (full NxN matrix)  ");
  multi_sw_set_name(sw, 1, "pp (packed NxN matrix)");


  X_c = full_c_create(N2, N2);
  Y_c = full_c_create(N2, N2);
  A_c = full_c_create(N2, N2);

  B_c  = full_c_create(N2, K2);
  B_c2 = full_c_create(N2, K2);

  A_ut_c = ut_c_create(N2, N2);


  for (i = 0; i < N2*N2; i++) {
    X_c->v_vector[i] = sin(cos(i));
  }

  /*
    printf("\n");
    printf("X_c:\n");
    full_c_printf(X_c);
  */

  full_c_set0(A_c);
  for (i = 0; i < N2; i++) {
    A_c->v[i][i] = i+1;
  }

  /*
    printf("\n");
    printf("A_c:\n");
    full_c_printf(A_c);
  */

  full_c_mmm(X_c, A_c, 0, Y_c);

  /*
    printf("\n");
    printf("Y_c:\n");
    full_c_printf(Y_c);
  */

  full_c_mmmT(Y_c, X_c, 0, A_c);

  for (i = 0; i < N2; i++) {
    A_c->v[i][i] += 1;
  }

  index = 0;
  for (i = 0; i < N2; i++) {
    for (j = 0; j < N2; j++) {
      if (i < j) {
	A_c->v[i][j] = 0;
      }
      else {
	A_ut_c->v[index] = A_c->v[i][j];
	index++;
      }
    }
  }


  /*
    printf("\n");
    printf("A_c:\n");
    full_c_printf(A_c);
  */

  multi_sw_start(sw, 0);
  r = epotrf(LAPACK_COL_MAJOR, 'U', N2, A_c->v_vector, A_c->m);
  multi_sw_stop(sw, 0);
  assert(r == 0);

  /*
    printf("\n");
    printf("chol(A):\n");
    full_c_printf(A_c);
  */

  /*
    printf("\n");
    printf("A_ut_c:\n");
    ut_c_printf(A_ut_c);
  */

  multi_sw_start(sw, 1);
  r = epptrf(LAPACK_COL_MAJOR, 'U', A_ut_c->n, A_ut_c->v);
  multi_sw_stop(sw, 1);
  assert(r == 0);

  /*
    printf("\n");
    printf("chol(A_ut_c):\n");
    ut_c_printf(A_ut_c);
  */

  for (i = 0; i < N2*K2; i++) {
    B_c->v_vector[i] = i + 1;
  }

  /*
    printf("\n");
    printf("B_c:\n");
    full_c_printf(B_c);
  */

  multi_sw_start(sw, 0);
  r = epotrs(LAPACK_COL_MAJOR, 'U', N2, K2, A_c->v_vector,
	     A_c->n, B_c->v_vector, N2);
  multi_sw_stop(sw, 0);
  assert(r == 0);

  /*
    printf("\n");
    printf("A_c\\B_c:\n");
    full_c_printf(B_c);
  */

  for (i = 0; i < N2*K2; i++) {
    B_c2->v_vector[i] = i + 1;
  }

  multi_sw_start(sw, 1);
  r = epptrs(LAPACK_COL_MAJOR, 'U', N2, K2, A_ut_c->v,
	     B_c2->v_vector, N2);
  multi_sw_stop(sw, 1);
  assert(r == 0);

  eaxpy(B_c->n*B_c->m, -1, B_c->v_vector, 1, B_c2->v_vector, 1);

  printf("\n");
  printf("||B_c - B_c2||_2 = ");
  printf_elem(enrm2(B_c2->n*B_c2->m, B_c2->v_vector, 1)
	      /enrm2(B_c->n*B_c->m, B_c->v_vector, 1));
  printf("\n");

  /*
    printf("\n");
    printf("A_ut_c\\B_c:\n");
    full_c_printf(B_c2);
  */

  printf("\n");
  multi_sw_printf(sw);

  full_c_destroy(&X_c);
  full_c_destroy(&Y_c);
  full_c_destroy(&A_c);
  full_c_destroy(&B_c);
  full_c_destroy(&B_c2);

  ut_c_destroy(&A_ut_c);

  multi_sw_destroy(&sw);

  return 0;
}
