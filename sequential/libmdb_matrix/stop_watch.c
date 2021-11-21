#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "stop_watch.h"


/*************
 * stop_watch
 *************/

stop_watch *stop_watch_create(void) {
  stop_watch *s;

  s = malloc(sizeof(stop_watch));
  assert(s);

  s->sec = 0;
  s->usec = 0;
  
  return s;
}

void stop_watch_destroy(stop_watch **s) {
  assert(*s);

  free(*s);
  *s = NULL;
}

void stop_watch_reset(stop_watch *s) {
  assert(s);
  
  s->sec = 0;
  s->usec = 0;
}

void stop_watch_start(stop_watch *s) {
  assert(s);

  gettimeofday(&(s->start_time), NULL);
}

void stop_watch_stop(stop_watch *s) {
  struct timeval stop_time;
  
  assert(s);

  gettimeofday(&stop_time, NULL);

  s->sec += stop_time.tv_sec - s->start_time.tv_sec;
  s->usec += stop_time.tv_usec - s->start_time.tv_usec;

  while (s->usec < 0) {
    s->sec--;
    s->usec += 1e6;
  }

  while(s->usec >= 1e6) {
    s->sec++;
    s->usec -= 1e6;
  }
}

void stop_watch_printf(const stop_watch *s) {
  assert(s);

  printf("Elapsed time: %f s\n", stop_watch_elapsed(s));
}

double stop_watch_elapsed(const stop_watch *s) {
  assert(s);

  return s->sec + s->usec/1e6;
}



/*************
 * multi_sw
 *************/

multi_sw *multi_sw_create(int n) {
  multi_sw *s;
  int i;
    
  assert(n > 0);

  s = malloc(sizeof(multi_sw));
  assert(s);

  s->sw = malloc(sizeof(stop_watch *) * n);
  assert(s->sw);

  s->names = malloc(sizeof(char *) * n);
  assert(s->names);
  
  for (i = 0; i < n; i++) {
    s->sw[i] = stop_watch_create();
    s->names[i] = NULL;
  }

  s->n = n;
  
  return s;
}

void multi_sw_destroy(multi_sw **s) {
  int i;
  
  assert(s);
  assert(*s);

  for (i = 0; i < (*s)->n; i++) {
    stop_watch_destroy(&((*s)->sw[i]));
  }

  free((*s)->sw);
  free((*s)->names);
  
  free(*s);
  *s = NULL;
}

void multi_sw_reset(multi_sw *s, int i) {
  assert(s);
  assert(i >= 0 && i < s->n);

  stop_watch_reset(s->sw[i]);
}

void multi_sw_start(multi_sw *s, int i) {
  assert(s);
  assert(i >= 0 && i < s->n);

  stop_watch_start(s->sw[i]);
}

void multi_sw_stop(multi_sw *s, int i) {
  assert(s);
  assert(i >= 0 && i < s->n);

  stop_watch_stop(s->sw[i]);
}

void multi_sw_set_name(const multi_sw *s, const int i, const char *name) {
  assert(s);
  assert(i >= 0 && i < s->n);
  assert(name);

  s->names[i] = name;
}

void multi_sw_printf(const multi_sw *s) {
  int i;
  
  assert(s);

  for (i = 0; i < s->n; i++) {
    if (s->names[i] != NULL) {
      printf("%16s: %6.3f s\n", s->names[i],
	     multi_sw_elapsed(s, i));
    }
    else {
      printf("[%d]: Elapsed time: %f s\n", i, multi_sw_elapsed(s, i));
    }
  }
}

void multi_sw_printf_total(const multi_sw *s) {
  assert(s);

  printf("Total elapsed time: %6.3f s\n", multi_sw_total_elapsed(s));
}

double multi_sw_elapsed(const multi_sw *s, int i) {
  assert(s);
  assert(i >= 0 && i < s->n);

  return stop_watch_elapsed(s->sw[i]);
}

double multi_sw_total_elapsed(const multi_sw *s) {
  int i;
  double total;
  
  assert(s);

  total = 0;
  for (i = 0; i < s->n; i++) {
    total += stop_watch_elapsed(s->sw[i]);
  }
  
  return total;
}

/******************************************************************************
 * Utility functions
 *****************************************************************************/

void printf_hms(double s) {
  int h, m;
  
  h = 0;
  while (s >= 3600) {
    h++;
    s -= 3600;
  }

  m = 0;
  while (s >= 60) {
    m++;
    s -= 60;
  }

  if (h > 0) {
    printf("%dh ", h);
  }

  if (m > 0) {
    printf("%dm ", m);
  }

  if (s > 0) {
    printf("%.2fs", s);
  }
  else {
    printf("<1s");
  }
}


int sprintf_hms(char *buffer, double s) {
  int h, m;
  int index;
  
  assert(buffer);
  
  h = 0;
  while (s >= 3600) {
    h++;
    s -= 3600;
  }

  m = 0;
  while (s >= 60) {
    m++;
    s -= 60;
  }

  index = 0;

  if (h > 0) {
    index += sprintf(buffer, "%dh %dm %.2fs", h, m, s);
  }
  else if (m > 0) {
    index += sprintf(buffer, "%dm %.2fs", m, s);
  }
  else if (s > 0) {
    index += sprintf(buffer, "%.2fs", s);
  }
  else {
    index += sprintf(buffer, "<1s");
  }

  return index;
}
