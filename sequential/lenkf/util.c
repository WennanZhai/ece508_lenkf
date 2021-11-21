#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <sys/utsname.h>
#include <unistd.h>

#ifdef OSX
#include <mach/task.h>
#include <mach/mach_init.h>
#endif

#include "util.h"


/*static struct timeval tic_time;*/
struct timeval tic_time;


void tic() {
  gettimeofday(&tic_time, NULL);
}


void toc() {
  struct timeval toc_time;
  struct timeval diff;

  gettimeofday(&toc_time, NULL);
  
  diff.tv_sec = toc_time.tv_sec - tic_time.tv_sec;
  diff.tv_usec = toc_time.tv_usec - tic_time.tv_usec;

  if (diff.tv_usec < 0) {
    diff.tv_sec--;
    diff.tv_usec += 1e6;
  }
  
  if (diff.tv_sec == 0) {
    printf("elapsed time: %d ms\n", (int) diff.tv_usec/1000);
  }
  else {
    double s;

    s = diff.tv_sec + diff.tv_usec/1e6;
    
    printf("elapsed time: %.2f s\n", s);
  }
}


int int_cmp(const void *v1, const void *v2) {
  const int *i1, *i2;

  i1 = (const int *) v1;
  i2 = (const int *) v2;

  if (i1 == i2) {
    return 0;
  }
  else {
    return i1 < i2;
  }
}


/* Returns the resident size of the self process - the actual physical
   memory the process is consuming (measured in megabytes) */
double get_res(void) {
#ifdef LINUX
  struct utsname system_info;
  FILE *self_statm_fid;
  unsigned int virt, res;
  int r;
  
  uname(&system_info);

  r = strcmp(system_info.sysname, "Linux");
  assert(r == 0);
  
  self_statm_fid = fopen("/proc/self/statm", "r");
  assert(self_statm_fid);

  r = fscanf(self_statm_fid, "%u %u", &virt, &res);
  assert(r == 2);
  
  fclose(self_statm_fid);

  /* Assumiung i386 - pages are 4 kB */
  return(((double)res*4)/1024);
#elif defined OSX
  /* THIS DOES NOT SEEM TO WORK */
  /* http://miknight.blogspot.com/2005/11/resident-set-size-in-mac-os-x.html */
  
  task_t task = MACH_PORT_NULL;
  struct task_basic_info t_info;
  mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;

  if (task_for_pid(current_task(), getpid(), &task) != KERN_SUCCESS) {
    abort();
  }
  
  task_info(task, TASK_BASIC_INFO, (task_info_t)&t_info, &t_info_count);
  return t_info.resident_size/1024;
#else
#error Unrecognized OS
#endif
}
