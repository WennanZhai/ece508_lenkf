#ifndef EBLAS_H
#define EBLAS_H

/* This is a bit of a kludge. The GSL library includes its own set of
 * C BLAS definitions. This prevents attempting to redefine these
 * definitions. */
#ifndef CBLAS_INDEX
#include <cblas.h>
#endif

#include "elem.h"


/*****************************************************************************/
/* Level 1 */

void erotg(elem *a, elem *b, elem *c, elem *s);
void erot(const int n, elem *x, const int incx,
          elem *y, const int incy, const elem c, const elem s);

void erotmg(elem *d1, elem *d2, elem *b1, const elem b2, elem *P);
void erotm(const int n, elem *x, const int incx,
           elem *y, const int incy, elem *P);

elem edot(const int n, const elem *x, const int incx,
          const elem *y, const int incy);

void eaxpy(int n, elem alpha, elem *x, int incx, elem *y, int incy);
elem enrm2(int n, elem *x, int incx);
elem easum(int n, elem *x, int incx);
int eiamax(int n, elem *x, int incx);
void escal(const int n, const elem alpha, elem *x, const int incx);

/* NOTE: The physical memory space of x and y cannot intersect for
   ecopy and eswap.  The behavior in this case is undefined as far as
   I can tell and I have observed segfaults as a result. */
void ecopy(const int n, const elem *x, const int incx,
           elem *y,  const int incy);
void eswap(const int n, elem *x, const int incx,
           elem *y,  const int incy);


/*****************************************************************************/
/* Level 2 */

void egemv(const enum CBLAS_ORDER order,
           const enum CBLAS_TRANSPOSE transa, const int m, const int n,
           const elem alpha, const elem *A, const int lda,
           const elem *x, const int incx, const elem beta,
           elem *y, const int incy);

void eger(const enum CBLAS_ORDER order, const int m, const int n,
          const elem alpha, const elem *x, const int incx,
          const elem *y, const int incy, elem *A, const int lda);

void etbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
	   const enum CBLAS_TRANSPOSE transa, const enum CBLAS_DIAG diag,
	   const int n, const int k, const elem *A, const int lda,
	   elem *x, const int incx);

void esyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
                const int n, const float alpha, const elem *x,
                const int incx, const elem *y, const int incy, elem *A,
                const int lda);


/*****************************************************************************/
/* Level 3 */

void egemm(const enum CBLAS_ORDER order,
           const enum CBLAS_TRANSPOSE transa,
           const enum CBLAS_TRANSPOSE transb,
           const int m, const int n, const int k, const elem alpha,
           const elem *A, const int lda,
           const elem *B, const int ldb,
           const elem beta, elem *C, const int ldc);


void etrmm(enum CBLAS_ORDER order, enum CBLAS_SIDE side,
           enum CBLAS_UPLO uplo, enum CBLAS_TRANSPOSE transa,
           enum CBLAS_DIAG diag, int m, int n, elem alpha,
           elem *A, int lda, elem *B, int ldb);



/*****************************************************************************/
/* BLAS routines for complex elements (z_elem) */

/*****************************************************************************/
/* Level 1 */

void zescal(const int n, const z_elem alpha, z_elem *x, const int incx);
void zeaxpy(const int n, const z_elem alpha, const z_elem *x, const int incx,
	    z_elem *y, const int incy);

/*****************************************************************************/
/* Level 2 */

void zetbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO uplo,
	    const enum CBLAS_TRANSPOSE transa, const enum CBLAS_DIAG diag,
	    const int n, const int k, const z_elem *A, const int lda,
	    z_elem *x, const int incx);


/*****************************************************************************/
/* Level 3 */

void zegemm(const enum CBLAS_ORDER order,
	    const enum CBLAS_TRANSPOSE transa,
	    const enum CBLAS_TRANSPOSE transb,
	    const int m, const int n, const int k, const z_elem *alpha,
	    const z_elem *A, const int lda,
	    const z_elem *B, const int ldb,
	    const z_elem *beta, z_elem *C, const int ldc);


#endif
