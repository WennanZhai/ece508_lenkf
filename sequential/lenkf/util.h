#ifndef UTIL_H
#define UTIL_H

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define SQR(x) ((x)*(x))
#define SGN(x) ((x) > 0 ? 1 : ((x) < 0 ? -1 : 0))

int int_cmp(const void *v1, const void *v2);


void tic(void);
void toc(void);

/* Returns the resident size of the self process - the actual physical
   memory the process is consuming (measured in megabytes) */
double get_res(void);

#endif
