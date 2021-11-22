#ifndef COUNTER_H
#define COUNTER_H

typedef struct {
  int n;
  int *c;
  const int *max;
} counter;

counter *counter_create(int n, const int *max);
void counter_destroy(counter **c);
void counter_destroy_keep_max(counter **c);
void counter_tick(counter *c);
void counter_tick_digit(counter *c, int n);
void counter_tick_digit_reset(counter *c, int n);
void counter_reset(counter *c);
void counter_printf(counter *c);
void counter_max_printf(counter *c);

#endif
