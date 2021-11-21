#include "eblas.h"


/*****************************************************************************/
/* Level 1 BLAS */

void eaxpy(int n, elem alpha, elem *x, int incx, elem *y, int incy) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_daxpy(n, alpha, x, incx, y, incy);
#elif defined ACML
  daxpy(n, alpha, x, incx, y, incy);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_saxpy(n, alpha, x, incx, y, incy);
#elif defined ACML
  saxpy(n, alpha, x, incx, y, incy);
#else
#error ??
#endif
#else
#error ?
#endif
}

elem enrm2(int n, elem *x, int incx) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  return cblas_dnrm2(n, x, incx);
#elif defined ACML
  return dnrm2(n, x, incx);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  return cblas_snrm2(n, x, incx);
#elif defined ACML
  return snrm2(n, x, incx);
#else
#error ??
#endif
#else
#error ?
#endif
}

elem easum(int n, elem *x, int incx) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  return cblas_dasum(n, x, incx);
#elif defined ACML
  return dasum(n, x, incx);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  return cblas_sasum(n, x, incx);
#elif defined ACML
  return sasum(n, x, incx);
#else
#error ??
#endif
#else
#error ?
#endif
}

int eiamax(int n, elem *x, int incx) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  return cblas_idamax(n, x, incx);
#elif defined ACML
  return idamax(n, x, incx);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  return cblas_isamax(n, x, incx);
#elif defined ACML
  return isamax(n, x, incx);
#else
#error ??
#endif
#else
#error ?
#endif
}

void erotmg(elem *d1, elem *d2, elem *b1, elem b2, elem *P) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_drotmg(d1, d2, b1, b2, P);
#elif defined ACML
  drotmg(*d1, *d2, *b1, b2, P);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_srotmg(d1, d2, b1, b2, P);
#elif defined ACML
  srotmg(*d1, *d2, *b1, b2, P);
#else
#error ??
#endif
#else
#error ?
#endif
}

void erotm(const int n, elem *x, const int incx,
	   elem *y, const int incy, elem *P) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_drotm(n, x, incx, y, incy, P);
#elif defined ACML
  drotm(n, x, incx, y, incy, P);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_srotm(n, x, incx, y, incy, P);
#elif defined ACML
  srotm(n, x, incx, y, incy, P);
#else
#error ??
#endif
#else
#error ?
#endif
}

void erotg(elem *a, elem *b, elem *c, elem *s) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_drotg(a, b, c, s);
#elif defined ACML
  drotg(a, b, c, s);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_srotg(a, b, c, s);
#elif defined ACML
  srotg(a, b, c, s);
#else
#error ??
#endif
#else
#error ?
#endif
}

void erot(const int n, elem *x, const int incx,
	  elem *y, const int incy, const elem c, const elem s) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_drot(n, x, incx, y, incy, c, s);
#elif defined ACML
  drot(n, x, incx, y, incy, c, s);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_srot(n, x, incx, y, incy, c, s);
#elif defined ACML
  srot(n, x, incx, y, incy, c, s);
#else
#error ??
#endif
#else
#error ?
#endif
}

elem edot(const int n, const elem *x, const int incx,
	  const elem *y, const int incy) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  return cblas_ddot(n, x, incx, y, incy);
#elif defined ACML
  return ddot(n, x, incx, y, incy);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  return cblas_sdot(n, x, incx, y, incy);
#elif defined ACML
  return sdot(n, x, incx, y, incy);
#else
#error ??
#endif
#else
#error ?
#endif
}

void escal(const int n, const elem alpha, elem *x, const int incx) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_dscal(n, alpha, x, incx);
#elif defined ACML
  dscal(n, alpha, x, incx);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_sscal(n, alpha, x, incx);
#elif defined ACML
  sscal(n, alpha, x, incx);
#else
#error ??
#endif
#else
#error ?
#endif
}

void ecopy(const int n, const elem *x, const int incx,
	   elem *y, const int incy) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_dcopy(n, x, incx, y, incy);
#elif defined ACML
  dcopy(n, x, incx, y, incy);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_scopy(n, x, incx, y, incy);
#elif defined ACML
  scopy(n, x, incx, y, incy);
#else
#error ??
#endif
#else
#error ?
#endif
}

void eswap(const int n, elem *x, const int incx,
	   elem *y, const int incy) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_dswap(n, x, incx, y, incy);
#elif defined ACML
  dswap(n, x, incx, y, incy);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_sswap(n, x, incx, y, incy);
#elif defined ACML
  sswap(n, x, incx, y, incy);
#else
#error ??
#endif
#else
#error ?
#endif
}


/*****************************************************************************/
/* Level 2 BLAS */

void egemv(const enum CBLAS_ORDER order,
	   const enum CBLAS_TRANSPOSE transa, const int m, const int n,
	   const elem alpha, const elem *A, const int lda,
	   const elem *x, const int incx, const elem beta,
	   elem *y, const int incy) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_dgemv(order, transa, m, n, alpha, A, lda, x, incx, beta, y, incy);
#elif defined ACML
  dgemv(order, transa, m, n, alpha, A, lda, x, incx, beta, y, incy);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_sgemv(order, transa, m, n, alpha, A, lda, x, incx, beta, y, incy);
#elif defined ACML
  sgemv(transa, m, n, alpha, A, lda, x, incx, beta, y, incy);
#else
#error ??
#endif
#else
#error ?
#endif
}

void eger(const enum CBLAS_ORDER order, const int m, const int n,
	  const elem alpha, const elem *x, const int incx,
	  const elem *y, const int incy, elem *A, const int lda) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_dger(order, m, n, alpha, x, incx, y, incy, A, lda);
#elif defined ACML
  dger(order, m, n, alpha, x, incx, y, incy, A, lda);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_sger(order, m, n, alpha, x, incx, y, incy, A, lda);
#elif defined ACML
  sger(m, n, alpha, x, incx, y, incy, A, lda);
#else
#error ??
#endif
#else
#error ?
#endif
}

void etbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
	   const enum CBLAS_TRANSPOSE transa, const enum CBLAS_DIAG diag,
	   const int n, const int k, const elem *A, const int lda,
	   elem *x, const int incx) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_dtbmv(order, uplo, transa, diag, n, k, A, lda, x, incx);
#elif defined ACML
  dtbmv(order, uplo, transa, diag, n, k, A, lda, x, incx);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_stbmv(order, uplo, transa, diag, n, k, A, lda, x, incx);
#elif defined ACML
  stbmv(order, uplo, transa, diag, n, k, A, lda, x, incx);
#else
#error ??
#endif
#else
#error ?
#endif
}

void esyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
	   const int n, const float alpha, const elem *x,
	   const int incx, const elem *y, const int incy, elem *A,
	   const int lda) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_dsyr2(order, uplo, n, alpha, x, incx, y, incy, A, lda);
#elif defined ACML
  dsyr2(order, uplo, n, alpha, x, incx, y, incy, A, lda);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_ssyr2(order, uplo, n, alpha, x, incx, y, incy, A, lda);
#elif defined ACML
  ssyr2(order, uplo, n, alpha, x, incx, y, incy, A, lda);
#else
#error ??
#endif
#else
#error ?
#endif
}


/*****************************************************************************/
/* Level 3 BLAS */

void egemm(const enum CBLAS_ORDER order,
	   const enum CBLAS_TRANSPOSE transa,
	   const enum CBLAS_TRANSPOSE transb, const int m, const int n,
	   const int k, const elem alpha, const elem *A,
	   const int lda, const elem *B, const int ldb,
	   const elem beta, elem *C, const int ldc) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_dgemm(order, transa, transb, m, n, k, alpha,
              A, lda, B, ldb, beta, C, ldc);
#elif defined ACML
  dgemm(order, transa, transb, m, n, k, alpha,
        A, lda, B, ldb, beta, C, ldc);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
   cblas_sgemm(order, transa, transb, m, n, k, alpha,
               A, lda, B, ldb, beta, C, ldc);
#elif defined ACML
    sgemm(transa, transb, m, n, k, alpha,
          A, lda, B, ldb, beta, C, ldc);
#else
#error ??
#endif
#else
#error ?
#endif
}

void etrmm(enum CBLAS_ORDER order, enum CBLAS_SIDE side,
	   enum CBLAS_UPLO uplo, enum CBLAS_TRANSPOSE transa,
	   enum CBLAS_DIAG diag, int m, int n, elem alpha,
	   elem *A, int lda, elem *B, int ldb) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_dtrmm(order, side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
#elif defined ACML
  dtrmm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_strmm(order, side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
#elif defined ACML
  strmm(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
#else
#error ??
#endif
#else
#error ?
#endif
}



/*****************************************************************************/
/* BLAS routines for complex elements (z_elem) */


/*****************************************************************************/
/* Level 1 */

void zescal(const int n, const z_elem alpha, z_elem *x, const int incx) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_zscal(n, alpha, x, incx);
#elif defined ACML
  zscal(n, alpha, x, incx);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_cscal(n, alpha, x, incx);
#elif defined ACML
  cscal(n, alpha, x, incx);
#else
#error ??
#endif
#else
#error ?
#endif
}


void zeaxpy(const int n, const z_elem alpha, const z_elem *x, const int incx,
	    z_elem *y, const int incy) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_zaxpy(n, alpha, x, incx, y, incy);
#elif defined ACML
  zaxpy(n, alpha, x, incx, y, incy);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_caxpy(n, alpha, x, incx, y, incy);
#elif defined ACML
  caxpy(n, alpha, x, incx, y, incy);
#else
#error ??
#endif
#else
#error ?
#endif
}


/*****************************************************************************/
/* Level 2 */

void zetbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
	    const enum CBLAS_TRANSPOSE transa, const enum CBLAS_DIAG diag,
	    const int n, const int k, const z_elem *A, const int lda,
	    z_elem *x, const int incx) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_ztbmv(order, uplo, transa, diag, n, k, A, lda, x, incx);
#elif defined ACML
  ztbmv(order, uplo, transa, diag, n, k, A, lda, x, incx);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_ctbmv(order, uplo, transa, diag, n, k, A, lda, x, incx);
#elif defined ACML
  ctbmv(order, uplo, transa, diag, n, k, A, lda, x, incx);
#else
#error ??
#endif
#else
#error ?
#endif
}


/*****************************************************************************/
/* Level 3 */

void zegemm(const enum CBLAS_ORDER order,
	    const enum CBLAS_TRANSPOSE transa,
	    const enum CBLAS_TRANSPOSE transb,
	    const int m, const int n, const int k, const z_elem *alpha,
	    const z_elem *A, const int lda,
	    const z_elem *B, const int ldb,
	    const z_elem *beta, z_elem *C, const int ldc) {
#ifdef DOUBLE_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
  cblas_zgemm(order, transa, transb, m, n, k, alpha,
              A, lda, B, ldb, beta, C, ldc);
#elif defined ACML
  zgemm(order, transa, transb, m, n, k, alpha,
        A, lda, B, ldb, beta, C, ldc);
#else
#error ??
#endif
#elif defined FLOAT_ELEM
#if defined (OPENBLAS) || defined (VECLIB)
   cblas_cgemm(order, transa, transb, m, n, k, alpha,
               A, lda, B, ldb, beta, C, ldc);
#elif defined ACML
    cgemm(transa, transb, m, n, k, alpha,
          A, lda, B, ldb, beta, C, ldc);
#else
#error ??
#endif
#else
#error ?
#endif
}
