#include <assert.h>
#include <stdio.h>

#include "elapack.h"


int eposv(const int order, const char uplo,
	   const int N, const int NRHS,
	   elem *A, const int lda,
	   elem *B, const int ldb) {
#ifdef VECLIB
  int info;
  char uplo_ = uplo;
  int N_ = N;
  int NRHS_ = NRHS;
  int lda_ = lda;
  int ldb_ = ldb;

  assert(order == LAPACK_COL_MAJOR);
#ifdef DOUBLE_ELEM
  assert(dposv_(&uplo_, &N_, &NRHS_, A, &lda_, B, &ldb_, &info) == 0);
#elif defined FLOAT_ELEM
  assert(sposv_(&uplo_, &N_, &NRHS_, A, &lda_, B, &ldb_, &info) == 0);
#else
#error ?
#endif
  return info;
#else
#ifdef DOUBLE_ELEM
  return LAPACKE_dposv(order, uplo, N, NRHS, A, lda, B, ldb);
#elif defined FLOAT_ELEM
  return LAPACKE_sposv(order, uplo, N, NRHS, A, lda, B, ldb);
#else
#error ?
#endif
#endif
}


int epotrf(const int order, const char uplo,
	   const int N, elem *A, const int lda) {
#ifdef VECLIB
  int info;
  char uplo_ = uplo;
  int N_ = N;
  int lda_ = lda;

  assert(order == LAPACK_COL_MAJOR);
#ifdef DOUBLE_ELEM
  assert(dpotrf_(&uplo_, &N_, A, &lda_, &info)  == 0);
#elif defined FLOAT_ELEM
  assert(spotrf_(&uplo_, &N_, A, &lda_, &info) == 0);
#else
#error ?
#endif
  return info;
#else
#ifdef DOUBLE_ELEM
  return LAPACKE_dpotrf(order, uplo, N, A, lda);
#elif defined FLOAT_ELEM
  return LAPACKE_spotrf(order, uplo, N, A, lda);
#else
#error ?
#endif
#endif
}


int epotrs(const int order, const char uplo,
	   const int N, const int NRHS,
	   elem *A, const int lda, elem *B, const int ldb) {
#ifdef VECLIB
  int info;
  char uplo_ = uplo;
  int N_ = N;
  int NRHS_ = NRHS;
  int lda_ = lda;
  int ldb_ = ldb;

  assert(order == LAPACK_COL_MAJOR);
#ifdef DOUBLE_ELEM
  assert(dpotrs_(&uplo_, &N_, &NRHS_, A, &lda_, B, &ldb_, &info) == 0);
#elif defined FLOAT_ELEM
  assert(spotrs_(&uplo_, &N_, &NRHS_, A, &lda_, B, &ldb_, &info) == 0);
#else
#error ?
#endif
  return info;
#else
#ifdef DOUBLE_ELEM
  return LAPACKE_dpotrs(order, uplo, N, NRHS, A, lda, B, ldb);
#elif defined FLOAT_ELEM
  return LAPACKE_spotrs(order, uplo, N, NRHS, A, lda, B, ldb);
#else
#error ?
#endif
#endif
}


int eppsv(const int order, const char uplo,
	  const int N, const int NRHS,
	  elem *A, elem *B, const int ldb) {
#ifdef VECLIB
  int info;
  char uplo_ = uplo;
  int N_ = N;
  int NRHS_ = NRHS;
  int ldb_ = ldb;

  assert(order == LAPACK_COL_MAJOR);
#ifdef DOUBLE_ELEM
  assert(dppsv_(&uplo_, &N_, &NRHS_, A, B, &ldb_, &info) == 0);
#elif defined FLOAT_ELEM
  assert(sppsv_(&uplo_, &N_, &NRHS_, A, B, &ldb_, &info) == 0);
#else
#error ?
#endif
  return info;
#else
#ifdef DOUBLE_ELEM
    return LAPACKE_dppsv(order, uplo, N, NRHS, A, B, ldb);
#elif FLOAT_ELEM
    return LAPACKE_sppsv(order, uplo, N, NRHS, A, B, ldb);
#else
#error ?
#endif
#endif
}


int epptrf(const int order, const char uplo,
	   const int N, elem *A) {
#ifdef VECLIB
  int info;
  char uplo_ = uplo;
  int N_ = N;

  assert(order == LAPACK_COL_MAJOR);
#ifdef DOUBLE_ELEM
  assert(dpptrf_(&uplo_, &N_, A, &info) == 0);
#elif defined FLOAT_ELEM
  assert(spptrf_(&uplo_, &N_, A, &info) == 0);
#else
#error ?
#endif
  return info;
#else
#ifdef DOUBLE_ELEM
    return LAPACKE_dpptrf(order, uplo, N, A);
#elif FLOAT_ELEM
    return LAPACKE_spptrf(order, uplo, N, A);
#else
#error ?
#endif
#endif
}


int epptrs(const int order, const char uplo,
	   const int N, const int NRHS,
	   elem *A, elem *B, const int ldb) {
#ifdef VECLIB
  int info;
  char uplo_ = uplo;
  int N_ = N;
  int NRHS_ = NRHS;
  int ldb_ = ldb;

  assert(order == LAPACK_COL_MAJOR);
#ifdef DOUBLE_ELEM
  assert(dpptrs_(&uplo_, &N_, &NRHS_, A, B, &ldb_, &info) == 0);
#elif defined FLOAT_ELEM
  assert(spptrs_(&uplo_, &N_, &NRHS_, A, B, &ldb_, &info) == 0);
#else
#error ?
#endif
  return info;
#else
#ifdef DOUBLE_ELEM
    return LAPACKE_dpptrs(order, uplo, N, NRHS, A, B, ldb);
#elif FLOAT_ELEM
    return LAPACKE_spptrs(order, uplo, N, NRHS, A, B, ldb);
#else
#error ?
#endif
#endif
}
