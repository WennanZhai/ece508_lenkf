#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "filter_new.h"
#include "eblas.h"
#include "vector.h"



/***********
 * c_filter_new
 ***********/

static void c_fprintf_row(FILE *fid, fftwe_complex *h, int c);

static fftwe_complex *c_filter_new_printf_r(c_filter_new *v, fftwe_complex *h,
					    int depth, ivector *count);


c_filter_new *c_filter_new_create(int rank, const int *n) {
  c_filter_new *v;
  int i;

  assert(rank > 0);

  v = malloc(sizeof(c_filter_new));
  assert(v);

  v->rank = rank;
  v->n = n;

  v->N = 1;
  for (i = 0; i < rank; i++) {
    v->N *= n[i];
  }

  v->h = fftwe_malloc(v->N * sizeof(fftw_complex));
  assert(v->h);

  v->d = S;

  v->forward_plan = fftwe_plan_dft(rank, n, v->h, v->h,
				   FFTW_FORWARD, FFTW_ESTIMATE);

  v->backward_plan = fftwe_plan_dft(rank, n, v->h, v->h,
				    FFTW_BACKWARD, FFTW_ESTIMATE);

  return v;
}

void c_filter_new_destroy(c_filter_new **v) {
  assert(v);
  assert(*v);

  fftwe_free((*v)->h);
  fftwe_destroy_plan((*v)->forward_plan);
  fftwe_destroy_plan((*v)->backward_plan);
  free((void *) (*v)->n);
  free(*v);

  *v = NULL;
}

void c_filter_new_dft(c_filter_new *v) {
  assert(v);

  fftwe_execute(v->forward_plan);
  v->d = fftwe_domain_flip(v->d);
}

void c_filter_new_idft(c_filter_new *v) {
  assert(v);
  assert(v->d == F);

  fftwe_execute(v->backward_plan);
  escal(2*v->N, ((elem) 1)/v->N, (elem *) v->h, 1);
  v->d = fftwe_domain_flip(v->d);
}

void c_filter_new_printf(c_filter_new *v) {
  ivector *count;

  assert(v);

  if (v->rank > 1) {
    count = ivector_create(v->rank - 1);
    ivector_set0(count);
  }
  else {
    count = NULL;
  }

  c_filter_new_printf_r(v, v->h, 0, count);

  if (v->rank > 1) {
    ivector_destroy(&count);
  }
  else {
    assert(count == NULL);
  }
}

static fftwe_complex *c_filter_new_printf_r(c_filter_new *v, fftwe_complex *h,
					    int depth, ivector *count) {
  int i;

  assert(v);
  assert(depth >= 0);
  assert(depth < v->rank);
  if (v->rank > 1) {
    assert(count);
  }
  assert(h);


  if(depth == v->rank - 1) {
    c_fprintf_row(stdout, h, v->n[v->rank - 1]);
    h += v->n[v->rank - 1];
  }
  else {
    if (depth == v->rank - 2 && v->rank > 2) {
      printf("(");
      for (i = 0; i < v->rank - 2; i++) {
	printf("%d, ", count->v[i]);
      }
      printf(":, :)\n");
    }

    for(i = 0; i < v->n[depth]; i++) {
      h = c_filter_new_printf_r(v, h, depth+1, count);
      count->v[depth]++;
    }
    count->v[depth] = 0;

    if (depth == v->rank - 2 && v->rank > 2) {
      printf("\n");
    }
  }

  return h;
}

static void c_fprintf_row(FILE *fid, fftwe_complex *h, int c) {
  int i;

  assert(h);
  assert(c > 0);

  for (i = 0; i < c; i++) {
    if (i == c - 1) {
      fftwe_complex_fprintf(fid, (const fftwe_complex *) &h[i]);
    }
    else {
      fftwe_complex_fprintf_s(fid, (const fftwe_complex *) &h[i]);
    }
  }
  fprintf(fid, "\n");
}

void c_filter_new_set_real0(c_filter_new *v) {
  assert(v);

  escal(v->N, 0, &(v->h[0][0]), 2);
}

void c_filter_new_set_imag0(c_filter_new *v) {
  assert(v);

  escal(v->N, 0, &(v->h[0][1]), 2);
}

void c_filter_new_set0(c_filter_new *v) {
  assert(v);

  escal(2*v->N, 0, &(v->h[0][0]), 1);
}



/******************************************************************************/

/***********
 * r_filter_new
 ***********/


static elem *r_filter_new_fprintf_r(FILE *fid, r_filter_new *v, elem *h,
				    int depth, ivector *count);

static void r_fprintf_row(FILE *fid, elem *h, int c);

static elem *r_filter_new_fprintf_dft_r(FILE *fid, r_filter_new *v, elem *h,
					int depth, ivector *count);

static int r_filter_row_major_idx(int rank, const int *i, const int *n_log);

static elem *r_filter_new_import_r(r_filter_new *v, elem *h, FILE *fid,
				   int depth);

static elem *r_filter_new_export_r(const r_filter_new *v, elem *h, FILE *fid,
				   int depth);


r_filter_new *r_filter_new_create(int rank, const int *n_log) {
  r_filter_new *v;
  int i;
  int *N_log;
  int *N_phy;

  assert(rank > 0);

  v = malloc(sizeof(r_filter_new));
  assert(v);

  v->rank = rank;

  v->n_log = n_log;

  N_log = malloc(rank * sizeof(int));
  assert(N_log);

  N_log[rank - 1] = n_log[rank - 1];
  for (i = rank - 2; i >= 0; i--) {
    N_log[i] = n_log[i] * N_log[i+1];
  }
  v->N_log = N_log;


  N_phy = malloc(rank * sizeof(int));
  assert(N_phy);

  N_phy[rank - 1] = 2*(n_log[rank - 1] / 2 + 1);
  for (i = rank - 2; i >= 0; i--) {
    N_phy[i] = n_log[i] * N_phy[i+1];
  }
  v->N_phy = N_phy;

  v->h = fftwe_malloc(v->N_phy[0] * sizeof(fftwe_complex));
  assert(v->h);

  v->d = S;

  v->forward_plan = fftwe_plan_dft_r2c(rank, n_log, v->h,
				       (fftwe_complex *) v->h, FFTW_ESTIMATE);

  v->backward_plan = fftwe_plan_dft_c2r(rank, n_log, (fftwe_complex *) v->h,
				       v->h, FFTW_ESTIMATE);

  return v;
}


r_filter_new *r_filter_new_create_same_dim(const r_filter_new *A) {
  int rank;
  int *n_log;
  int i;

  assert(A);

  rank = A->rank;

  n_log = malloc(rank * sizeof(int));
  assert(n_log);

  for (i = 0; i < rank; i++) {
    n_log[i] = A->n_log[i];
  }

  return r_filter_new_create(rank, n_log);
}


void r_filter_new_destroy(r_filter_new **v) {
  assert(v);
  assert(*v);

  fftwe_free((*v)->h);
  free((void *) (*v)->n_log);
  free((void *) (*v)->N_log);
  free((void *) (*v)->N_phy);
  fftwe_destroy_plan((*v)->forward_plan);
  fftwe_destroy_plan((*v)->backward_plan);
  free(*v);

  *v = NULL;
}


void r_filter_new_dft(r_filter_new *v) {
  assert(v);

  fftwe_execute(v->forward_plan);
  v->d = fftwe_domain_flip(v->d);
}


void r_filter_new_idft(r_filter_new *v) {
  assert(v);
  assert(v->d == F);

  fftwe_execute(v->backward_plan);
  escal(2*v->N_phy[0], ((elem) 1)/v->N_log[0], (elem *) v->h, 1);
  v->d = fftwe_domain_flip(v->d);
}


void r_filter_new_set0(r_filter_new *v) {
  assert(v);

  set0(v->h, v->N_phy[0]);
}


void r_filter_new_printf(r_filter_new *v) {
  r_filter_new_fprintf(stdout, v);
}


void r_filter_new_fprintf(FILE *fid, r_filter_new *v) {
  ivector *count;

  assert(v);

  if (v->rank > 1) {
    count = ivector_create(v->rank - 1);
    ivector_set0(count);
  }
  else {
    count = NULL;
  }

  r_filter_new_fprintf_r(fid, v, v->h, 0, count);

  if (v->rank > 1) {
    ivector_destroy(&count);
  }
  else {
    assert(count == NULL);
  }
}


static elem *r_filter_new_fprintf_r(FILE *fid, r_filter_new *v, elem *h,
				    int depth, ivector *count) {
  int i;

  assert(v);
  assert(depth >= 0);
  assert(depth < v->rank);
  if (v->rank > 1) {
    assert(count);
  }
  assert(h);


  if (depth == v->rank - 1) {
    if (v->d == S) {
      r_fprintf_row(fid, h, v->n_log[v->rank - 1]);
    }
    else if (v->d == F) {
      c_fprintf_row(fid, (fftwe_complex *) h,
		    floor(v->n_log[v->rank - 1]/2) + 1);
    }
    else {
      assert(0);
    }

    h += 2*((int) (floor(v->n_log[v->rank - 1]/2) + 1));
  }
  else {
    if (depth == v->rank - 2 && v->rank > 2) {
      fprintf(fid, "(");
      for (i = 0; i < v->rank - 2; i++) {
	fprintf(fid, "%d, ", count->v[i]);
      }
      fprintf(fid, ":, :)\n");
    }

    for(i = 0; i < v->n_log[depth]; i++) {
      h = r_filter_new_fprintf_r(fid, v, h, depth+1, count);
      count->v[depth]++;
    }
    count->v[depth] = 0;

    if (depth == v->rank - 2 && v->rank > 2) {
      fprintf(fid, "\n");
    }
  }

  return h;
}


static void r_fprintf_row(FILE *fid, elem *h, int c) {
  int i;

  assert(h);
  assert(c > 0);

  for (i = 0; i < c; i++) {
    if (i == c - 1) {
      fprintf_elem(fid, h[i]);
    }
    else {
      fprintf_elem_s(fid, h[i]);
    }
  }
  fprintf(fid, "\n");
}


void r_filter_new_printf_dft(r_filter_new *v) {
  r_filter_new_fprintf_dft(stdout, v);
}


void r_filter_new_fprintf_dft(FILE *fid, r_filter_new *v) {
 ivector *count;

 assert(v);
 assert(v->d == F);

 if (v->rank > 1) {
   count = ivector_create(v->rank - 1);
   ivector_set0(count);
 }
 else {
   count = NULL;
 }

 r_filter_new_fprintf_dft_r(fid, v, v->h, 0, count);

 if (v->rank > 1) {
   ivector_destroy(&count);
 }
 else {
   assert(count == NULL);
 }
}


static elem *r_filter_new_fprintf_dft_r(FILE *fid, r_filter_new *v, elem *h,
					int depth, ivector *count) {
  int i, k, k_half, idx;
  int *k_idx;
  z_elem h_conj;
  z_elem *h_ptr;

  assert(v);
  assert(depth >= 0);
  assert(depth < v->rank);
  if (v->rank > 1) {
    assert(count);
  }
  assert(h);


  if (depth == v->rank - 1) {
    k_half = floor(v->n_log[v->rank-1]/2);

    for (k = 0; k < k_half + 1; k++) {
      fprintf_z_elem_s(fid, (const z_elem *) h);
      h += 2;
    }

    k_idx = malloc(v->rank * sizeof(int));
    assert(k_idx);

    for (i = 0; i < v->rank - 1; i++) {
      k_idx[i] = (v->n_log[i] - count->v[i]) % v->n_log[i];
    }

    for (k = k_half + 1; k < v->n_log[v->rank - 1]; k++) {
      k_idx[v->rank - 1] = (v->n_log[v->rank - 1] - k) % v->n_log[v->rank - 1];

      idx = r_filter_row_major_idx(v->rank, k_idx, v->n_log);

      h_ptr = (z_elem *) v->h;
      h_ptr += idx;
      h_conj[0] = (*h_ptr)[0];
      h_conj[1] = -(*h_ptr)[1];
      fprintf_z_elem_s(fid, (const z_elem *) &h_conj);
    }

    free(k_idx);

    fprintf(fid, "\n");
  }
  else {
    if (depth == v->rank - 2 && v->rank > 2) {
      fprintf(fid, "(");
      for (i = 0; i < v->rank - 2; i++) {
	fprintf(fid, "%d, ", count->v[i]);
      }
      fprintf(fid, ":, :)\n");
    }

    for(i = 0; i < v->n_log[depth]; i++) {
      h = r_filter_new_fprintf_dft_r(fid, v, h, depth+1, count);
      count->v[depth]++;
    }
    count->v[depth] = 0;

    if (v->rank - depth == 2 && v->rank > 2) {
      fprintf(fid, "\n");
    }
  }

  return h;
}


static int r_filter_row_major_idx(int rank, const int *i, const int *n_log) {
  int j, idx;

  assert(rank > 0);
  assert(i);
  assert(n_log);

  idx = i[0];
  for (j = 1; j < rank; j++) {
    if (j == rank - 1) {
      idx *= floor(n_log[j]/2) + 1;
    }
    else {
      idx *= n_log[j];
    }

    idx += i[j];
  }

  return idx;
}


void r_filter_new_execute_r(const r_filter_new *h, r_filter_new *x) {
  int i;

  assert(h);
  assert(x);
  assert(h->d == F);
  assert(x->d == F);
  assert(h->rank == x->rank);

  for (i = 0; i < h->rank; i++) {
    assert(h->n_log[i] == x->n_log[i]);
  }

  zetbmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, h->N_phy[0], 0,
	 (const z_elem *) h->h, 1, (z_elem *) x->h, 1);
}


r_filter_new *r_filter_new_import(const char *filename) {
  FILE *fid;
  int rank;
  int *n_log;
  int r;
  int sizeof_elem;
  r_filter_new *v;

  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  r = fread(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);
  assert(sizeof_elem == sizeof(elem));

  r = fread(&rank, sizeof(int), 1, fid);
  assert(r == 1);

  assert(rank > 0);
  n_log = malloc(rank * sizeof(int));
  assert(n_log);

  r = fread(n_log, sizeof(int), rank, fid);
  assert(r == rank);

  v = r_filter_new_create(rank, n_log);

  r_filter_new_import_r(v, v->h, fid, 0);

  fclose(fid);

  return v;
}


static elem *r_filter_new_import_r(r_filter_new *v, elem *h, FILE *fid,
				   int depth) {
  int i;
  int r;

  assert(v);
  assert(depth >= 0);
  assert(depth < v->rank);
  assert(h);
  assert(fid);


  if (depth == v->rank - 1) {
    r = fread(h, sizeof(elem), v->n_log[v->rank - 1], fid);
    h += 2*((int) (floor(v->n_log[v->rank - 1]/2) + 1));
  }
  else {
    for(i = 0; i < v->n_log[depth]; i++) {
      h = r_filter_new_import_r(v, h, fid, depth+1);
    }
  }

  return h;
}


void r_filter_new_export(const char *filename, const r_filter_new *v) {
  FILE *fid;
  int r;
  int sizeof_elem;

  assert(filename);
  assert(v);
  assert(v->d == S);

  fid = fopen(filename, "w");
  assert(fid);

  sizeof_elem = sizeof(elem);
  r = fwrite(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&v->rank, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(v->n_log, sizeof(int), v->rank, fid);
  assert(r == v->rank);

  r_filter_new_export_r(v, v->h, fid, 0);

  fclose(fid);
}


static elem *r_filter_new_export_r(const r_filter_new *v, elem *h, FILE *fid,
				   int depth) {
  int i;
  int r;

  assert(v);
  assert(depth >= 0);
  assert(depth < v->rank);
  assert(h);
  assert(fid);


  if (depth == v->rank - 1) {
    r = fwrite(h, sizeof(elem), v->n_log[v->rank - 1], fid);
    h += 2*((int) (floor(v->n_log[v->rank - 1]/2) + 1));
  }
  else {
    for(i = 0; i < v->n_log[depth]; i++) {
      h = r_filter_new_export_r(v, h, fid, depth+1);
    }
  }

  return h;
}


elem *r_filter_new_get_ptr(const r_filter_new *v, const int *n) {
  int i;
  elem *e_ptr;
  z_elem *ze_ptr;
  int rank;

  assert(v);
  assert(n);

  rank = v->rank;

  e_ptr = v->h;
  for (i = 0; i < v->rank - 1; i++) {
    assert(n[i] >= 0);
    assert(n[i] < v->n_log[i]);

    e_ptr += n[i] * v->N_phy[i+1];
  }

  assert(n[rank - 1] >= 0);
  if (v->d == S) {
    assert(n[rank - 1] < v->n_log[i]);
    e_ptr += n[rank - 1];
  }
  else if (v->d == F) {
    assert(n[rank - 1] < floor(v->n_log[v->rank - 1]/2) + 1);
    ze_ptr = (z_elem *) e_ptr;
    e_ptr = &(ze_ptr[n[rank - 1]][0]);
  }
  else {
    assert(0);
  }

  return e_ptr;
}


void r_filter_new_scal(r_filter_new *v, elem alpha) {
  assert(v);

  escal(v->N_phy[0], alpha, v->h, 1);
}



/******************************************************************************/

/************
 * rs2_filter_new
 *************/


static elem *rs2_filter_new_printf_r(rs2_filter_new *v, elem *h, int depth,
				     ivector *count);

static void rs2_filter_new_printf_dft_r(rs2_filter_new *v, int depth,
					ivector *k);

static int rs2_filter_row_major_idx(int rank, const int *i, const int *n_phy);

static elem *rs2_filter_new_execute_r_r(const rs2_filter_new *v,
					r_filter_new *x, elem *x_h,
					int depth, ivector *k);



rs2_filter_new *rs2_filter_new_create(int rank, const int *n_phy) {
  rs2_filter_new *v;
  int i;
  fftw_r2r_kind *kind;

  assert(rank > 0);
  assert(n_phy);

  v = malloc(sizeof(rs2_filter_new));
  assert(v);

  v->rank = rank;

  v->n_phy = n_phy;

  v->N_phy = 1;
  for (i = 0; i < rank; i++) {
    v->N_phy *= n_phy[i];
  }

  v->h = malloc(v->N_phy * sizeof(elem));
  assert(v->h);

  v->N_log = pow(2, v->rank)*v->N_phy;

  v->d = S;

  kind = malloc(rank * sizeof(fftw_r2r_kind));
  assert(kind);

  for (i = 0; i < rank; i++) {
    kind[i] = FFTW_REDFT10;
  }

  v->forward_plan = fftwe_plan_r2r(rank, n_phy, v->h, v->h, kind,
				   FFTW_ESTIMATE);

  for (i = 0; i < rank; i++) {
    kind[i] = FFTW_REDFT01;
  }

  v->backward_plan = fftwe_plan_r2r(rank, n_phy, v->h, v->h, kind,
				    FFTW_ESTIMATE);

  free(kind);

  return v;
}


void rs2_filter_new_destroy(rs2_filter_new **v) {
  assert(v);
  assert(*v);

  fftwe_free((*v)->h);
  free((void *) (*v)->n_phy);
  fftwe_destroy_plan((*v)->forward_plan);
  fftwe_destroy_plan((*v)->backward_plan);
  free(*v);

  *v = NULL;
}


rs2_filter_new *rs2_filter_new_import(char *filename) {
  FILE *fid;
  int rank;
  int *n_phy;
  int r;
  int sizeof_elem;
  rs2_filter_new *v;

  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  r = fread(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);
  assert(sizeof_elem == sizeof(elem));

  r = fread(&rank, sizeof(int), 1, fid);
  assert(r == 1);

  assert(rank > 0);

  n_phy = malloc(rank * sizeof(int));
  assert(n_phy);

  r = fread(n_phy, sizeof(int), rank, fid);
  assert(r == rank);

  v = rs2_filter_new_create(rank, n_phy);

  r = fread(v->h, sizeof(elem), v->N_phy, fid);
  assert(r == v->N_phy);

  fclose(fid);

  return v;
}


void rs2_filter_new_export(char *filename, rs2_filter_new *v) {
  FILE *fid;
  int r, sizeof_elem;

  fid = fopen(filename, "w");
  assert(fid);

  sizeof_elem = sizeof(elem);
  r = fwrite(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&v->rank, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(v->n_phy, sizeof(int), v->rank, fid);
  assert(r == v->rank);

  r = fwrite(v->h, sizeof(elem), v->N_phy, fid);
  assert(r == v->N_phy);

  fclose(fid);
}


void rs2_filter_new_set0(rs2_filter_new *v) {
  assert(v);

  escal(v->N_phy, 0, v->h, 1);
}


void rs2_filter_new_dft(rs2_filter_new *v) {
  assert(v);

  fftwe_execute(v->forward_plan);
  v->d = fftwe_domain_flip(v->d);
}


void rs2_filter_new_idft(rs2_filter_new *v) {
  assert(v);
  assert(v->d == F);

  fftwe_execute(v->backward_plan);
  escal(v->N_phy, ((elem) 1)/v->N_log, (elem *) v->h, 1);
  v->d = fftwe_domain_flip(v->d);
}


void rs2_filter_new_printf(rs2_filter_new *v) {
  ivector *count;

  assert(v);

  if (v->rank > 1) {
    count = ivector_create(v->rank - 1);
    ivector_set0(count);
  }
  else {
    count = NULL;
  }

  rs2_filter_new_printf_r(v, v->h, 0, count);

  if (v->rank > 1) {
    ivector_destroy(&count);
  }
  else {
    assert(count == NULL);
  }
}

static elem *rs2_filter_new_printf_r(rs2_filter_new *v, elem *h, int depth,
				     ivector *count) {
  int i;

  assert(v);
  assert(depth >= 0);
  assert(depth < v->rank);
  if (v->rank > 1) {
    assert(count);
  }
  assert(h);


  if(depth == v->rank - 1) {
    r_fprintf_row(stdout, h, v->n_phy[v->rank - 1]);
    h += v->n_phy[v->rank - 1];
  }
  else {
    if (depth == v->rank - 2 && v->rank > 2) {
      printf("(");
      for (i = 0; i < v->rank - 2; i++) {
	printf("%d, ", count->v[i]);
      }
      printf(":, :)\n");
    }

    for(i = 0; i < v->n_phy[depth]; i++) {
      h = rs2_filter_new_printf_r(v, h, depth+1, count);
      count->v[depth]++;
    }
    count->v[depth] = 0;

    if (v->rank - depth == 2 && v->rank > 2) {
      printf("\n");
    }
  }

  return h;
}


void rs2_filter_new_printf_dft(rs2_filter_new *v) {
  ivector *k;

  assert(v);
  assert(v->d == F);

  k = ivector_create(v->rank);
  ivector_set0(k);

  rs2_filter_new_printf_dft_r(v, 0, k);

  ivector_destroy(&k);
}


static void rs2_filter_new_printf_dft_r(rs2_filter_new *v, int depth,
					ivector *k) {
  int i;
  z_elem H_k;

  assert(v);
  assert(depth >= 0);
  assert(depth < v->rank);
  assert(k);


  if(depth == v->rank - 1) {
    for (i = 0; i < 2*v->n_phy[v->rank - 1]; i++) {
      k->v[v->rank - 1] = i;
      rs2_filter_new_dft_get(v, k, &H_k);
      printf_z_elem_s((const z_elem *) &H_k);
    }
    printf("\n");
  }
  else {
    if (depth == v->rank - 2 && v->rank > 2) {
      printf("(");
      for (i = 0; i < v->rank - 2; i++) {
	printf("%d, ", k->v[i]);
      }
      printf(":, :)\n");
    }

    for(i = 0; i < 2*v->n_phy[depth]; i++) {
      rs2_filter_new_printf_dft_r(v, depth+1, k);
      k->v[depth]++;
    }
    k->v[depth] = 0;

    if (depth == v->rank - 2 && v->rank > 2) {
      printf("\n");
    }
  }
}


void rs2_filter_new_dft_get(const rs2_filter_new *v, ivector *k, z_elem *H_k) {
  z_elem scale_factor, z;
  elem theta;

  int *k_idx;
  int i, idx;

  assert(v);
  assert(k);
  assert(H_k);
  assert(v->rank == k->n);

  k_idx = malloc(v->rank * sizeof(int));
  assert(k_idx);

  scale_factor[0] = 1;
  scale_factor[1] = 0;

  for (i = 0; i < v->rank; i++) {
    /* See O&S2 (8.174) */
    if (k->v[i] == 0) {
      k_idx[i] = k->v[i];
    }
    else if (k->v[i] >= 1 && k->v[i] <= v->n_phy[i] - 1) {
      theta = M_PI*k->v[i]/((elem) 2 * v->n_phy[i]);
      expj(theta, &z);
      zmul(&scale_factor, (const z_elem *) &z);
      k_idx[i] = k->v[i];
    }
    else if (k->v[i] == v->n_phy[i]) {
      (*H_k)[0] = 0;
      (*H_k)[1] = 0;

      free(k_idx);
      return;
    }
    else if (k->v[i] >= v->n_phy[i] + 1 && k->v[i] <= 2*v->n_phy[i] - 1) {
      theta = M_PI*k->v[i]/((elem) 2 * v->n_phy[i]);
      expj(theta, &z);
      z[0] = -z[0];
      z[1] = -z[1];
      zmul(&scale_factor, (const z_elem *) &z);
      k_idx[i] = 2*v->n_phy[i] - k->v[i];
    }
    else {
      assert(0);
    }
  }

  idx = rs2_filter_row_major_idx(v->rank, k_idx, v->n_phy);

  (*H_k)[0] = v->h[idx];
  (*H_k)[1] = 0;

  zmul(H_k, (const z_elem *) &scale_factor);

  free(k_idx);
}


static int rs2_filter_row_major_idx(int rank, const int *i, const int *n_phy) {
  int j, idx;

  assert(rank > 0);
  assert(i);
  assert(n_phy);

  idx = i[0];
  for (j = 1; j < rank; j++) {
    idx *= n_phy[j];
    idx += i[j];
  }

  return idx;
}


void rs2_filter_new_execute_r(const rs2_filter_new *v, r_filter_new *x) {
  int i;
  ivector *k;

  assert(v);
  assert(x);
  assert(v->d == F);
  assert(x->d == F);
  assert(v->rank == x->rank);

  for (i = 0; i < v->rank; i++) {
    assert(x->n_log[i] == 2*v->n_phy[i]);
  }

  k = ivector_create(v->rank);
  ivector_set0(k);

  rs2_filter_new_execute_r_r(v, x, x->h, 0, k);

  ivector_destroy(&k);
}


static elem *rs2_filter_new_execute_r_r(const rs2_filter_new *v,
					r_filter_new *x, elem *x_h,
					int depth, ivector *k) {
  int i;
  z_elem H_k;

  assert(v);
  assert(depth >= 0);
  assert(depth < v->rank);
  assert(x);
  assert(k);


  if(depth == v->rank - 1) {
    for (i = 0; i < floor(x->n_log[v->rank - 1]/2) + 1; i++) {
      k->v[v->rank - 1] = i;
      rs2_filter_new_dft_get(v, k, &H_k);

      zmul((z_elem *) x_h, (const z_elem *) &H_k);
      x_h += 2;
    }
  }
  else {
    for(i = 0; i < 2*v->n_phy[depth]; i++) {
      x_h = rs2_filter_new_execute_r_r(v, x, x_h, depth+1, k);
      k->v[depth]++;
    }
    k->v[depth] = 0;
  }

  return x_h;
}



/******************************************************************************/

/************
 * rs12_filter_new
 *************/


static elem *rs12_filter_new_printf_r(rs12_filter_new *v, elem *h, int depth,
				     ivector *count);

static int rs12_filter_row_major_idx(int rank, const int *i, const int *n_phy);


static void rs12_filter_new_printf_dft_r(rs12_filter_new *v, int depth,
					ivector *k);

static elem *rs12_filter_new_execute_r_r(const rs12_filter_new *v,
					r_filter_new *x, elem *x_h,
					int depth, ivector *k);


rs12_filter_new *rs12_filter_new_create(int rank, const int *n_phy,
					const rs12_filter_new_kind *kind) {
  rs12_filter_new *v;
  int *n_log;
  int i;
  fftw_r2r_kind *kind_fftw;

  assert(rank > 0);
  assert(n_phy);
  assert(kind);

  v = malloc(sizeof(rs12_filter_new));
  assert(v);

  v->rank = rank;
  v->n_phy = n_phy;
  v->kind = kind;

  v->N_phy = 1;
  for (i = 0; i < rank; i++) {
    /* Singleton dimensions are not allowed! */
    assert(n_phy[i] > 1);
    v->N_phy *= n_phy[i];
  }

  v->h = malloc(v->N_phy * sizeof(elem));
  assert(v->h);

  n_log = malloc(rank * sizeof(int));
  assert(n_log);

  for (i = 0; i < rank; i++) {
    if (kind[i] == TYPE1) {
      n_log[i] = 2*(n_phy[i] - 1);
    }
    else if (kind[i] == TYPE2) {
      n_log[i] = 2*n_phy[i];
    }
    else {
      assert(0);
    }
  }

  v->n_log = n_log;

  v->N_log = 1;
  for (i = 0; i < rank; i++) {
    v->N_log *= n_log[i];
  }

  v->d = S;

  kind_fftw = malloc(rank * sizeof(fftw_r2r_kind));
  assert(kind_fftw);


  for (i = 0; i < rank; i++) {
    if (kind[i] == TYPE1) {
      kind_fftw[i] = FFTW_REDFT00;
    }
    else if (kind[i] == TYPE2) {
      kind_fftw[i] = FFTW_REDFT10;
    }
    else {
      assert(0);
    }
  }

  v->forward_plan = fftwe_plan_r2r(rank, n_phy, v->h, v->h, kind_fftw,
				   FFTW_ESTIMATE);

  for (i = 0; i < rank; i++) {
    if (kind[i] == TYPE1) {
      kind_fftw[i] = FFTW_REDFT00;
    }
    else if (kind[i] == TYPE2) {
      kind_fftw[i] = FFTW_REDFT01;
    }
    else {
      assert(0);
    }
  }

  v->backward_plan = fftwe_plan_r2r(rank, n_phy, v->h, v->h, kind_fftw,
				    FFTW_ESTIMATE);

  free(kind_fftw);

  return v;
}


void rs12_filter_new_destroy(rs12_filter_new **v) {
  assert(v);
  assert(*v);

  fftwe_free((*v)->h);
  free((void *) (*v)->n_phy);
  free((void *) (*v)->n_log);
  free((void *) (*v)->kind);
  fftwe_destroy_plan((*v)->forward_plan);
  fftwe_destroy_plan((*v)->backward_plan);
  free(*v);

  *v = NULL;
}


void rs12_filter_new_set0(rs12_filter_new *v) {
  assert(v);

  escal(v->N_phy, 0, v->h, 1);
}


void rs12_filter_new_dft(rs12_filter_new *v) {
  assert(v);

  fftwe_execute(v->forward_plan);
  v->d = fftwe_domain_flip(v->d);
}


void rs12_filter_new_idft(rs12_filter_new *v) {
  assert(v);
  assert(v->d == F);

  fftwe_execute(v->backward_plan);
  escal(v->N_phy, ((elem) 1)/v->N_log, (elem *) v->h, 1);
  v->d = fftwe_domain_flip(v->d);
}


void rs12_filter_new_printf(rs12_filter_new *v) {
  ivector *count;

  assert(v);

  if (v->rank > 1) {
    count = ivector_create(v->rank - 1);
    ivector_set0(count);
  }
  else {
    count = NULL;
  }

  rs12_filter_new_printf_r(v, v->h, 0, count);

  if (v->rank > 1) {
    ivector_destroy(&count);
  }
  else {
    assert(count == NULL);
  }
}

static elem *rs12_filter_new_printf_r(rs12_filter_new *v, elem *h, int depth,
				     ivector *count) {
  int i;

  assert(v);
  assert(depth >= 0);
  assert(depth < v->rank);
  if (v->rank > 1) {
    assert(count);
  }
  assert(h);


  if(depth == v->rank - 1) {
    r_fprintf_row(stdout, h, v->n_phy[v->rank - 1]);
    h += v->n_phy[v->rank - 1];
  }
  else {
    if (depth == v->rank - 2 && v->rank > 2) {
      printf("(");
      for (i = 0; i < v->rank - 2; i++) {
	printf("%d, ", count->v[i]);
      }
      printf(":, :)\n");
    }

    for(i = 0; i < v->n_phy[depth]; i++) {
      h = rs12_filter_new_printf_r(v, h, depth+1, count);
      count->v[depth]++;
    }
    count->v[depth] = 0;

    if (v->rank - depth == 2 && v->rank > 2) {
      printf("\n");
    }
  }

  return h;
}


static int rs12_filter_row_major_idx(int rank, const int *i, const int *n_phy) {
  int j, idx;

  assert(rank > 0);
  assert(i);
  assert(n_phy);

  idx = i[0];
  for (j = 1; j < rank; j++) {
    idx *= n_phy[j];
    idx += i[j];
  }

  return idx;
}


void rs12_filter_new_dft_get(const rs12_filter_new *v, ivector *k, z_elem *H_k) {
  z_elem scale_factor, z;
  elem theta;

  int *k_idx;
  int i, idx;

  assert(v);
  assert(k);
  assert(H_k);
  assert(v->rank == k->n);

  k_idx = malloc(v->rank * sizeof(int));
  assert(k_idx);

  scale_factor[0] = 1;
  scale_factor[1] = 0;

  for (i = 0; i < v->rank; i++) {
    switch (v->kind[i]) {
    case TYPE1:
      /* See O&S2 (8.164) */
      if (k->v[i] < v->n_phy[i]) {
	k_idx[i] = k->v[i];
      }
      else if (k->v[i] < v->n_log[i]) {
	k_idx[i] = v->n_log[i] - k->v[i];
      }
      else {
	assert(0);
      }
      break;
    case TYPE2:
      /* See O&S2 (8.174) */
      if (k->v[i] == 0) {
	k_idx[i] = k->v[i];
      }
      else if (k->v[i] >= 1 && k->v[i] <= v->n_phy[i] - 1) {
	theta = M_PI*k->v[i]/((elem) 2 * v->n_phy[i]);
	expj(theta, &z);
	zmul(&scale_factor, (const z_elem *) &z);
	k_idx[i] = k->v[i];
      }
      else if (k->v[i] == v->n_phy[i]) {
	(*H_k)[0] = 0;
	(*H_k)[1] = 0;

	free(k_idx);
	return;
      }
      else if (k->v[i] >= v->n_phy[i] + 1 && k->v[i] <= v->n_log[i]) {
	theta = M_PI*k->v[i]/((elem) 2 * v->n_phy[i]);
	expj(theta, &z);
	z[0] = -z[0];
	z[1] = -z[1];
	zmul(&scale_factor, (const z_elem *) &z);
	k_idx[i] = 2*v->n_phy[i] - k->v[i];
      }
      else {
	assert(0);
      }
      break;
    default:
      assert(0);
    }
  }

  idx = rs12_filter_row_major_idx(v->rank, k_idx, v->n_phy);

  (*H_k)[0] = v->h[idx];
  (*H_k)[1] = 0;

  zmul(H_k, (const z_elem *) &scale_factor);

  free(k_idx);
}


void rs12_filter_new_printf_dft(rs12_filter_new *v) {
  ivector *k;

  assert(v);
  assert(v->d == F);

  k = ivector_create(v->rank);
  ivector_set0(k);

  rs12_filter_new_printf_dft_r(v, 0, k);

  ivector_destroy(&k);
}


static void rs12_filter_new_printf_dft_r(rs12_filter_new *v, int depth,
					ivector *k) {
  int i;
  z_elem H_k;

  assert(v);
  assert(depth >= 0);
  assert(depth < v->rank);
  assert(k);


  if(depth == v->rank - 1) {
    for (i = 0; i < v->n_log[v->rank - 1]; i++) {
      k->v[v->rank - 1] = i;
      rs12_filter_new_dft_get(v, k, &H_k);
      printf_z_elem_s((const z_elem *) &H_k);
    }
    printf("\n");
  }
  else {
    if (depth == v->rank - 2 && v->rank > 2) {
      printf("(");
      for (i = 0; i < v->rank - 2; i++) {
	printf("%d, ", k->v[i]);
      }
      printf(":, :)\n");
    }

    for(i = 0; i < v->n_log[depth]; i++) {
      rs12_filter_new_printf_dft_r(v, depth+1, k);
      k->v[depth]++;
    }
    k->v[depth] = 0;

    if (depth == v->rank - 2 && v->rank > 2) {
      printf("\n");
    }
  }
}


rs12_filter_new *rs12_filter_new_import(char *filename) {
  FILE *fid;
  int rank;
  int *n_phy;
  int sizeof_elem;
  rs12_filter_new_kind *kind;

  int r;
  rs12_filter_new *v;

  assert(filename);

  fid = fopen(filename, "r");
  assert(fid);

  r = fread(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);
  assert(sizeof_elem == sizeof(elem));

  r = fread(&rank, sizeof(int), 1, fid);
  assert(r == 1);

  assert(rank > 0);

  n_phy = malloc(rank * sizeof(int));
  assert(n_phy);

  r = fread(n_phy, sizeof(int), rank, fid);
  assert(r == rank);

  kind = malloc(rank * sizeof(int));
  assert(kind);

  assert(sizeof(rs12_filter_new_kind) == sizeof(int));
  r = fread(kind, sizeof(rs12_filter_new_kind), rank, fid);
  assert(r == rank);

  v = rs12_filter_new_create(rank, n_phy, kind);

  r = fread(v->h, sizeof(elem), v->N_phy, fid);
  assert(r == v->N_phy);

  fclose(fid);

  return v;
}


void rs12_filter_new_export(char *filename, rs12_filter_new *v) {
  FILE *fid;
  int r, sizeof_elem;

  fid = fopen(filename, "w");
  assert(fid);

  sizeof_elem = sizeof(elem);
  r = fwrite(&sizeof_elem, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(&v->rank, sizeof(int), 1, fid);
  assert(r == 1);

  r = fwrite(v->n_phy, sizeof(int), v->rank, fid);
  assert(r == v->rank);

  assert(sizeof(rs12_filter_new_kind) == sizeof(int));
  r = fwrite(v->kind, sizeof(rs12_filter_new_kind), v->rank, fid);
  assert(r == v->rank);

  r = fwrite(v->h, sizeof(elem), v->N_phy, fid);
  assert(r == v->N_phy);

  fclose(fid);
}


void rs12_filter_new_execute_r(const rs12_filter_new *v, r_filter_new *x) {
  int i;
  ivector *k;

  assert(v);
  assert(x);
  assert(v->d == F);
  assert(x->d == F);
  assert(v->rank == x->rank);

  for (i = 0; i < v->rank; i++) {
    assert(x->n_log[i] == v->n_log[i]);
  }

  k = ivector_create(v->rank);
  ivector_set0(k);

  rs12_filter_new_execute_r_r(v, x, x->h, 0, k);

  ivector_destroy(&k);
}


static elem *rs12_filter_new_execute_r_r(const rs12_filter_new *v,
					r_filter_new *x, elem *x_h,
					int depth, ivector *k) {
  int i;
  z_elem H_k;

  assert(v);
  assert(depth >= 0);
  assert(depth < v->rank);
  assert(x);
  assert(k);


  if(depth == v->rank - 1) {
    for (i = 0; i < floor(x->n_log[v->rank - 1]/2) + 1; i++) {
      k->v[v->rank - 1] = i;
      rs12_filter_new_dft_get(v, k, &H_k);

      zmul((z_elem *) x_h, (const z_elem *) &H_k);
      x_h += 2;
    }
  }
  else {
    for(i = 0; i < v->n_log[depth]; i++) {
      x_h = rs12_filter_new_execute_r_r(v, x, x_h, depth+1, k);
      k->v[depth]++;
    }
    k->v[depth] = 0;
  }

  return x_h;
}
