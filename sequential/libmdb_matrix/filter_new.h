#ifndef FILTER_NEW_H
#define FILTER_NEW_H

#include "vector.h"
#include "elem.h"
#include "fftwe.h"


/***********
 * c_filter_new
 ***********/

/* A complex impulse response filter. */

typedef struct {
  fftwe_complex *h;

  int rank;
  const int *n;
  int N;
  
  domain d;
  
  fftwe_plan forward_plan;
  fftwe_plan backward_plan;
} c_filter_new;

/* h is stored in row-major as described in the fftw 3.1.2 manual as
   described on page 16.  This means that the last dimension of h
   varies most quickly and the first dimension most slowly.  Suppose h
   is a 2D 4x3 PSF (i.e., a 4 row and 3 column image).  Then, rank=2
   and n[0] = 4 and n[1] = 3 and h[0] is (0,0) and h[1] is (0,1),
   i.e., row 0, column 1. */

c_filter_new *c_filter_new_create(int rank, const int *n);
void c_filter_new_destroy(c_filter_new **v);
void c_filter_new_dft(c_filter_new *v);
void c_filter_new_idft(c_filter_new *v);
void c_filter_new_printf(c_filter_new *v);
void c_filter_new_set_real0(c_filter_new *v);
void c_filter_new_set_imag0(c_filter_new *v);
void c_filter_new_set0(c_filter_new *v);



/***********
 * r_filter_new
 ***********/

/* A real impulse response filter. */

typedef struct {
  elem *h;

  int rank;
  const int *n_log;
  const int *N_phy;
  const int *N_log;
  
  domain d;
  
  fftwe_plan forward_plan;
  fftwe_plan backward_plan;
} r_filter_new;

r_filter_new *r_filter_new_create(int rank, const int *n_log);
r_filter_new *r_filter_new_create_same_dim(const r_filter_new *A);
void r_filter_new_destroy(r_filter_new **v);
r_filter_new *r_filter_new_import(const char *filename);
void r_filter_new_export(const char *filename, const r_filter_new *v);
elem *r_filter_new_get_ptr(const r_filter_new *v, const int *n);
void r_filter_new_dft(r_filter_new *v);
void r_filter_new_idft(r_filter_new *v);
void r_filter_new_set0(r_filter_new *v);
void r_filter_new_printf(r_filter_new *v);
void r_filter_new_printf_dft(r_filter_new *v);
void r_filter_new_fprintf(FILE *fid, r_filter_new *v);
void r_filter_new_fprintf_dft(FILE *fid, r_filter_new *v);
void r_filter_new_execute_r(const r_filter_new *h, r_filter_new *x);
void r_filter_new_scal(r_filter_new *v, elem alpha);



/*************
 * rs2_filter_new
 *************/


/* A real, Type-II symmetric, even logical length impulse response,
 * multidimensional filter.
 *
 * Let length(h[n]) = n_log with n_log even.  The impulse response
 * h[n] is symmetric in the following sense: h[n] = h[n_log - n], 0 <=
 * n < n_log and the stored impulse response v->h = [a b c d] actually
 * represents the Type-II symmetric, even length impulse response:
 *
 *    h[n] = [a b c d d c b a].
 *
 * The inverse transform is the Type-III DCT.
 *
 * The relationship to the DFT in this case is (n_phy = length(v->h), and
 * length(h[n]) = 2*n_phy):
 *
 * H[k] = H_DCT[0],                k = 0
 *        exp(j*pi*k/(2*n_log)),   k = 1, ..., n_phy - 1
 *        0,                       k = n_phy
 *        -exp(j*pi*k/(2*n_log)),  k = n_phy + 1, ..., 2*n_phy - 1
 *
 * See (8.174), O&S2
 *
 */


typedef struct {
  elem *h;

  int rank;
  const int *n_phy;   /* physical dimensions of h[n_1,...,n_rank] under
			 Type-II symmetry - each logical dimension is twice
			 the physical dimension */
  int N_phy;
  int N_log;
  
  domain d;
  
  fftwe_plan forward_plan;
  fftwe_plan backward_plan;
} rs2_filter_new;


rs2_filter_new *rs2_filter_new_create(int rank, const int *n_phy);
void rs2_filter_new_destroy(rs2_filter_new **v);
rs2_filter_new *rs2_filter_new_import(char *filename);
void rs2_filter_new_export(char *filename, rs2_filter_new *v);
void rs2_filter_new_set0(rs2_filter_new *v);
void rs2_filter_new_dft(rs2_filter_new *v);
void rs2_filter_new_idft(rs2_filter_new *v);
void rs2_filter_new_printf(rs2_filter_new *v);
void rs2_filter_new_printf_dft(rs2_filter_new *v);
void rs2_filter_new_dft_get(const rs2_filter_new *v, ivector *k, z_elem *H_k);
void rs2_filter_new_execute_r(const rs2_filter_new *v, r_filter_new *x);



/*************
 * rs12_filter_new
 *************/

/* A real, Type-I or Type-II, symmetric, even logical length impulse
 * response, multidimensional filter.
 *
 * Each dimension of the filter may exhibit Type-I or Type-II even
 * symmetry.  The type of symmetry implies how the physically stored
 * filter coefficients relate to the logical, symmetric, even length
 * filter.
 *
 * Type-I even symmetry
 * --------------------
 *
 * Suppose that h_phy[dim] = [a b c d],
 * then         h_log[dim] = [a b c d c b].
 *
 * Note that the logical filter is of length 2*(n_phy[dim]-1).
 *
 * The frequency transform is DCT-I (what FFTW calls REDFT00).  Note
 * that DCT-1 does not exist for n_phy[dim] = 1! (Type-I symmetry is
 * undefined in this case)
 *
 * The inverse transform is DCT-I scaled by 1/2*(n_phy[dim] - 1).
 *
 * The DFT is related to the DCT-I by the following (see (8.164) from
 * O&S2):
 *
 * dft(h_log[dim])(k) =
 *  dct1(h_phy[dim])(k),                     k = 0, ..., n_phy[dim]-1
 *  dct1(h_phy[dim])(2*n_phy[dim] - 2 - k),  k = n_phy[dim], ..., 2*n_phy[dim]-3
 *
 * Note: n_log[dim] = 2*(n_phy[dim] - 1), so the range on k in the
 * second case above is k = n_phy[dim], ..., n_log[dim] - 1.
 *
 *
 * Type-II even symmetry
 * ---------------------
 *
 * Suppose that h_phy[dim] = [a b c d],
 * then         h_log[dim] = [a b c d d c b a].
 *
 * Note that the logical filter is of length 2*n_phy[dim].
 *
 * The frequency transform is DCT-II, i.e., the DCT.
 *
 * The inverse transform is DCT-III scaled by 1/2*n_phy[dim].
 *
 * The DFT is related to the DCT-III by the following (see (8.174)
 * from O&S2):
 *
 * dft(h_log[dim](k) =
 *  exp(j*pi*k/(2*n_phy[dim]))*dct2(h_phy[dim])(k),
 *        k = 0, ..., n_phy[dim]-1
 * -exp(j*pi*k/(2*n_phy[dim]))*dct2(h_phy[dim])(2*n_phy[dim] - k),
 *        k = n_phy[dim], ..., 2*n_phy[dim]-1
 *
 * Notes:
 * - n_log[dim] = 2*n_phy[dim], so the range on k in the second above
 *   case is k = n_phy[dim], ..., n_log[dim]-1.
 * - dft(h_log[dim](0)          = dct2(h_phy[dim])(k)
 * - dft(h_log[dim](n_phy[dim]) = 0
 *
 *
 * NOTE: Linear convolution with a real sequence is difficult!
 *
 * Consider a 1D problem.  Let x be of length n.  Suppose that h is
 * Type-II symmetric and has logical length n.  Then, the linear
 * convolution is of length l=n+m-1.  This implies that both x and h
 * must be zero-padded to at least length l to compute the linear
 * convolution using FFTs.  Suppose n=6 and h_phy = [a b c] which
 * implies that h_log = [a b c c b a].  Suppose m=4 and the length of
 * the linear convolution is then l=6+4-1 = 9.  We must zero-pad to
 * logical length 10 (the DCTs and DSTs implemented by FFTW only
 * operate on logical even length sequences).  Let h_phy = [c b a 0 0]
 * which implies that h_log = [c b a 0 0 0 0 a b c], which is Type-II
 * symmetric and corresponds to a circular shift of the original
 * filter and will result in a circular shift in the linear
 * convolution (which can be easily accounted for).
 *
 * The situation is much more complicated with Type-I symmetry.
 * Consider h_phy = [a b c d] which implies that h_log = [a b c d c b]
 * and the logical length is n = 6.  The linear convolution is still
 * of length l=6+4-1 = 10.  It is possible to simultaneously zero pad
 * h_phy and maintain Type-I symmetry by upsampling, i.e., consider
 * h_phy = [a 0 b 0 c 0 d], which then implies the logical filter
 * h_log = [a 0 b 0 c 0 d 0 c 0 b 0] which is length 12.  However, the
 * DFT spectrum of the filter h has now been compressed by a factor of
 * 2.  How then do we compute the linear convolution?  The following
 * references may give a means for implementing this idea:
 *
 * S. A. Martucci, "Symmetric Convolution and the Discrete Sine and
 * Cosine Transforms," IEEE Trans. Signal Proc., Vol. 42,
 * pp. 1038--1051, 1994.
 *
 * V. G. Reju, N. Koh, and I. Y. Soon, "Convolution Using Discrete
 * Sine and Cosine Transforms," IEEE Signal Proc. Letters, Vol. 14,
 * pp. 445--448, 2007.
 *
 */

typedef enum rs12_filter_new_kind_enum {TYPE1 = 1, TYPE2 = 2}
  rs12_filter_new_kind;

typedef struct {
  elem *h;

  int rank;
  const int *n_phy;
  const int *n_log;
  const rs12_filter_new_kind *kind;

  int N_phy;
  int N_log;
  
  domain d;
  
  fftwe_plan forward_plan;
  fftwe_plan backward_plan;
} rs12_filter_new;


rs12_filter_new *rs12_filter_new_create(int rank, const int *n_phy,
					const rs12_filter_new_kind *kind);
void rs12_filter_new_destroy(rs12_filter_new **v);
rs12_filter_new *rs12_filter_new_import(char *filename);
void rs12_filter_new_export(char *filename, rs12_filter_new *v);
void rs12_filter_new_set0(rs12_filter_new *v);
void rs12_filter_new_dft(rs12_filter_new *v);
void rs12_filter_new_idft(rs12_filter_new *v);
void rs12_filter_new_printf(rs12_filter_new *v);
void rs12_filter_new_printf_dft(rs12_filter_new *v);
void rs12_filter_new_dft_get(const rs12_filter_new *v, ivector *k, z_elem *H_k);
void rs12_filter_new_execute_r(const rs12_filter_new *v, r_filter_new *x);


#endif
