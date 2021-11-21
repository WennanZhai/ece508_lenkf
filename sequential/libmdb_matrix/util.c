#include <assert.h>
#include <math.h>

#include "util.h"


boolean is_even(int x) {
  return x % 2 == 0 ? True : False;
}


boolean is_odd(int x) {
  return x % 2 == 1 ? True : False;
}


boolean xnor(boolean x, boolean y) {
  return !(x ^ y);
}


elem linspace(elem start, elem end, int n, int i) {
  return start + i * (end - start) / (n - 1);
}


elem linspace_alt(elem start, elem end, int n, int i, boolean endpoint) {
  int m;

  if (endpoint) {
    m = n - 1;
  }
  else {
    m = n;
  }
  
  return start + i * (end - start) / m;
}


elem fftfreq(int n, int k) {
  elem f;
  
  assert(n > 0);

  f = linspace_alt(0, 1, n, k, False);
  if (f >= 0.5) {
    f -= 1;
  }
  return f;
}


int fftshift(int n, int k) {
  return (k + (int)(floor((n + 1)/2))) % n;
}
