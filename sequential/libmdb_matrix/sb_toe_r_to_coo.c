#include <assert.h>
#include <stdio.h>

#include "mdb_matrix_d.h"


int main(int argc, char **argv) {
  char *C_fname, *C_coo_fname;
  sb_toe_r *C;
  sparse_coo *C_coo;

  assert(argc == 3);

  C_fname = argv[1];
  C_coo_fname = argv[2];

  fprintf(stderr, "loading sb_toe_r matrix from %s\n", C_fname);
  C = sb_toe_r_import(C_fname);

  C_coo = sb_toe_r_convert_coo(C);

  fprintf(stderr, "exporting COO matrix to %s\n", C_coo_fname);
  sparse_coo_export(C_coo_fname, C_coo);

  sb_toe_r_destroy(&C);
  sparse_coo_destroy(&C_coo);

  return 0;
}
