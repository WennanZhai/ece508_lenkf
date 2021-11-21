#ifndef STOP_WATCH_H
#define STOP_WATCH_H

#include <sys/time.h>


typedef struct {
  struct timeval start_time;
  int sec;
  int usec;
} stop_watch;

stop_watch *stop_watch_create(void);
void stop_watch_destroy(stop_watch **s);
void stop_watch_reset(stop_watch *s);
void stop_watch_start(stop_watch *s);
void stop_watch_stop(stop_watch *s);
void stop_watch_printf(const stop_watch *s);
double stop_watch_elapsed(const stop_watch *s);


typedef struct {
  stop_watch **sw;
  const char **names;
  int n;
} multi_sw;

multi_sw *multi_sw_create(int n);
void multi_sw_destroy(multi_sw **s);
void multi_sw_reset(multi_sw *s, int i);
void multi_sw_start(multi_sw *s, int i);
void multi_sw_stop(multi_sw *s, int i);
void multi_sw_set_name(const multi_sw *s, const int i, const char *name);
void multi_sw_printf(const multi_sw *s);
void multi_sw_printf_total(const multi_sw *s);
double multi_sw_elapsed(const multi_sw *s, int i);
double multi_sw_total_elapsed(const multi_sw *s);

void printf_hms(double s);
int sprintf_hms(char *buffer, double s);

#endif
