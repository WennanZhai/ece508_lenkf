#ifndef ELAPACK_H
#define ELAPACK_H

#ifdef OPENBLAS
#include <lapacke.h>
#elif defined VECLIB
#include <clapack.h>
#define LAPACK_ROW_MAJOR               101
#define LAPACK_COL_MAJOR               102
#endif

#include "elem.h"


int eposv(const int order, const char uplo,
	  const int N, const int NRHS,
	  elem *A, const int lda,
	  elem *B, const int ldb);

int epotrf(const int order, const char uplo,
	   const int N, elem *A, const int lda);

int epotrs(const int order, const char uplo,
	   const int N, const int NRHS,
	   elem *A, const int lda, elem *B, const int ldb);


int eppsv(const int order, const char uplo,
	  const int N, const int NRHS,
	  elem *A, elem *B, const int ldb);

int epptrf(const int order, const char uplo,
	   const int N, elem *A);

int epptrs(const int order, const char uplo,
	   const int N, const int NRHS,
	   elem *A, elem *B, const int ldb);

#endif
