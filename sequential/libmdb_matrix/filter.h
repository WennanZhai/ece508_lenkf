#ifndef FILTER_H
#define FILTER_H

#include "elem.h"
#include "fftwe.h"


/***********
 * c_filter
 ***********/

/* A complex impulse response filter. */

typedef struct {
  fftwe_complex *h;
  int n;
  domain d;
  
  fftwe_plan forward_plan;
  fftwe_plan backward_plan;
} c_filter;

c_filter *c_filter_create(int n);
void c_filter_destroy(c_filter **v);
void c_filter_dft(c_filter *v);
void c_filter_idft(c_filter *v);
void c_filter_printf(c_filter *v);
void c_filter_set_real0(c_filter *v);
void c_filter_set_imag0(c_filter *v);
void c_filter_set0(c_filter *v);


/***********
 * r_filter
 ***********/

/* A real impulse response filter. */

typedef struct {
  elem *h;
  int n_phy;
  int n_log;
  domain d;
  
  fftwe_plan forward_plan;
  fftwe_plan backward_plan;
} r_filter;

r_filter *r_filter_create(int n);
void r_filter_destroy(r_filter **v);
void r_filter_dft(r_filter *v);
void r_filter_idft(r_filter *v);
void r_filter_printf(r_filter *v);
void r_filter_set0(r_filter *v);
void r_filter_dft_get(const r_filter *v, int k, z_elem *H_k);



/* The Type-I and Type-II DCT are discussed in 8.8 of Oppenhiem and
   Schafer, 2nd edition (henceforth O&S2).  The relationship between
   these transforms and the DFT are made explicit. */

/*************
 * rs1_filter
 *************/

/* A real, Type-I symmetric, even length impulse response filter.
 *
 * Let length(h[n]) = n_log with n_log even.  The impulse response
 * h[n] is symmetric in the following sense: h[n] = h[n_log - n], 0 <=
 * n < n_log and the stored impulse response v->h = [a b c d] actually
 * represents the Type-I symmetric, even length impulse response:
 *
 *    h[n] = [a b c d c b].
 *
 * The inverse transform is also Type-I DCT.
 *
 * The relationship to the DFT in this case is (n_phy = length(v->h), and
 * length(h[n]) = 2*(n_phy-1)):
 *
 * H[k] = H_DCT[k],           k = 0, ..., n_phy - 1
 *        H_DCT[2*k - 2 -l],  k = n_phy, ..., 2*n_phy - 3
 *
 * See (8.164), O&S2.
 *
 * Be wary of the note mentioned at the end of Section 2.5.2 of the
 * FFTW version 3.1 manual that mentions that FFTW_REDFT00 (Type-I
 * DCT) can be slow versus other implementations for the sake of
 * accuracy.  They recommend compiling hard-coded transforms via fftw
 * when n_phy is small (in the multidimensional case).
 *
 */

typedef struct {
  elem *h;
  int n_phy;       /* physical length of v->h */
  int n_log;       /* logical length of h[n] under Type-I symmetry */
  domain d;
  
  fftwe_plan plan;
} rs1_filter;

rs1_filter *rs1_filter_create(int n_phy);
rs1_filter *rs1_filter_import(char *filename);
rs1_filter *rs1_filter_import_zp(char *filename, int n_phy_zp);
void rs1_filter_destroy(rs1_filter **v);
void rs1_filter_dft(rs1_filter *v);
void rs1_filter_idft(rs1_filter *v);
void rs1_filter_printf(const rs1_filter *v);
void rs1_filter_set0(rs1_filter *v);
elem rs1_filter_dft_get(const rs1_filter *v, int k);
void rs1_filter_execute(const rs1_filter *v, c_filter *x);
void rs1_filter_execute_noblas(const rs1_filter *v, c_filter *x);



/*************
 * rs2_filter
 *************/

/* A real, Type-II symmetric, even length impulse response filter.
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
  int n_phy;   /* physical length of v->h */
  int n_log;   /* logical length of h[n] under Type-II symmetry */
  domain d;
  
  fftwe_plan forward_plan;
  fftwe_plan backward_plan;
} rs2_filter;

rs2_filter *rs2_filter_create(int n_phy);
rs2_filter *rs2_filter_import(char *filename);
rs2_filter *rs2_filter_import_zp(const char *filename, int n_phy, int n_phy_zp);
void rs2_filter_destroy(rs2_filter **v);
void rs2_filter_dft(rs2_filter *v);
void rs2_filter_idft(rs2_filter *v);
void rs2_filter_printf(const rs2_filter *v);
void rs2_filter_printf_dft(const rs2_filter *v);
void rs2_filter_set0(rs2_filter *v);
void rs2_filter_dft_get(const rs2_filter *v, int k, z_elem *H_k);
void rs2_filter_execute_c(const rs2_filter *v, c_filter *x);
void rs2_filter_execute_r(const rs2_filter *v, r_filter *x);


#endif
