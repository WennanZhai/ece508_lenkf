#ifndef UTIL_H
#define UTIL_H

#include <stdint.h>

#include "elem.h"
#include "boolean.h"


#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

/*
#ifdef LONG_PTR
#define PTR_XOR(x, y) (void *)(((long int)(x)) ^ ((long int)(y)))
#else
#define PTR_XOR(x, y) (void *)(((int)(x)) ^ ((int)(y)))
#endif
*/


#define PTR_XOR(x, y) (void *)(((uintptr_t)(x)) ^ ((uintptr_t)(y)))

boolean is_even(int x);
boolean is_odd(int x);

boolean xnor(boolean x, boolean y);

elem linspace(elem start, elem end, int n, int i);
elem linspace_alt(elem start, elem end, int n, int i, boolean endpoint);

elem fftfreq(int n, int k);
int fftshift(int n, int k);

#endif
