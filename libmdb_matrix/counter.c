#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "counter.h"


counter *counter_create(int n, const int *max) {
  counter *c;
  int i;
  
  assert(n > 0);
  assert(max);

  for (i = 0; i < n; i++) {
    assert(max[i] > 0);
  }
  
  c = malloc(sizeof(counter));
  assert(c);

  c->n = n;
  
  c->c = malloc(n * sizeof(int));
  assert(c->c);

  c->max = max;
  
  return c;
}


void counter_destroy(counter **c) {
  assert(c);
  assert(*c);

  free((*c)->c);
  free((int *) (*c)->max);

  free(*c);
  *c = NULL;
}


void counter_destroy_keep_max(counter **c) {
  assert(c);
  assert(*c);

  free((*c)->c);

  free(*c);
  *c = NULL;
}


void counter_tick(counter *c) {
  counter_tick_digit(c, c->n - 1);
}


void counter_tick_digit(counter *c, int n) {
  int i;
  
  assert(c);
  assert(n >= 0);
  assert(n < c->n);
  
  c->c[n]++;

  for (i = n; i >= 1; i--) {
    if (c->c[i] >= c->max[i]) {
      c->c[i-1] += c->max[i] - c->c[i] + 1;
      c->c[i] = 0;
    }
  }

  /* Overflow error */
  if (c->c[0] >= c->max[0]) {
    assert(0);
  }
}


void counter_tick_digit_reset(counter *c, int n) {
  int i;
  
  counter_tick_digit(c, n);

  for (i = n+1; i < c->n; i++) {
    c->c[i] = 0;
  }
}


void counter_reset(counter *c) {
  int i;
  
  assert(c);

  for (i = 0; i < c->n; i++) {
    c->c[i] = 0;
  }
}


void counter_printf(counter *c) {
  int i;
  
  assert(c);

  printf("[");
  for (i = 0; i < c->n; i++) {
    printf("%d ", c->c[i]);
  }
  printf("\b]");
}


void counter_max_printf(counter *c) {
  int i;
  
  assert(c);

  printf("[");
  for (i = 0; i < c->n; i++) {
    printf("%d ", c->max[i]);
  }
  printf("\b]");
}
