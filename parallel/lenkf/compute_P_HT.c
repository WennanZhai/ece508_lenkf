#include <stdlib.h>
#include <assert.h>

#include "ensemble.h"
#include "arg_bundle.h"
#include "lenkf.h"
#include "../kernel_code/compute.h"

void compute_P_HT(arg_bundle *ab, const sparse_rcs *H,
                  const int row_H, const char name_H,
                  const int n_rows);


void do_compute_P_HT(const char *P_HT_fname,
                     const char *e_fname,
                     const char *H_fname,
                     const char *C_fname,
                     const char *HT_fname,
                     const char *eT_fname) {
    //arg_bundle *ab;
    sparse_rcs *H,*HT;
    sparse_rcs *P_HT_rcs;
    int rank, i;
    ensemble *e, *eT;
    sparse_coo *C;

    // ab = malloc(sizeof(arg_bundle));
    //assert(ab);

    fprintf(stderr, "loading e from %s\n", e_fname);
    /*ab->*/e = ensemble_import(e_fname);
    fprintf(stderr, "N=%d  L=%d\n", e->N, e->L);

    fprintf(stderr, "\n");
    fprintf(stderr, "loading eT from %s\n", eT_fname);
    eT = ensemble_import(eT_fname);
    fprintf(stderr, "N=%d  L=%d\n", eT->N, eT->L);

    //ab->config = malloc(sizeof(lenkf_config));
    //assert(ab->config);
    //ab->config->N = ab->e->N;

    //ab->P_HT = NULL;

    fprintf(stderr, "\n");
    fprintf(stderr, "loading H from %s\n", H_fname);
    H = sparse_rcs_import(H_fname);
    fprintf(stderr, "m=%d  n=%d  N=%d\n", H->m, H->n, H->N);

    fprintf(stderr, "\n");
    fprintf(stderr, "loading HT from %s\n", HT_fname);
    HT = sparse_rcs_import(HT_fname);
    fprintf(stderr, "m=%d  n=%d  N=%d\n", HT->m, HT->n, HT->N);

    fprintf(stderr, "\n");
    fprintf(stderr, "loading C from %s\n", C_fname);
    //ab->C = sb_toe_r_import(C_fname);
    //ab->C_it = sb_toe_r_nz_it_create(ab->C);
    C = sb_toe_r_convert_coo(sb_toe_r_import(C_fname));
    //rank = ab->C->dim->rank;
    //fprintf(stderr, "rank=%d\n", rank);
    // fprintf(stderr, "n_phy=[ ");
    // for (i = 0; i < rank; i++) {
    //     fprintf(stderr, "%d ", ab->C->dim->n_phy[i]);
    // }
    // fprintf(stderr, "]\n");
    // fprintf(stderr, "n    =[ ");
    // for (i = 0; i < rank; i++) {
    //     fprintf(stderr, "%d ", ab->C->dim->n[i]);
    // }
    // fprintf(stderr, "]\n");
    // fprintf(stderr, "N_phy=[ ");
    // for (i = 0; i < rank; i++) {
    //     fprintf(stderr, "%d ", ab->C->dim->N_phy[i]);
    // }
    // fprintf(stderr, "]\n");
    // fprintf(stderr, "N    =[ ");
    // for (i = 0; i < rank; i++) {
    //     fprintf(stderr, "%d ", ab->C->dim->N[i]);
    // }
    // fprintf(stderr, "]\n");

    //compute_P_HT(ab, H, 0, 'H', H->m);

    critical_kernel_wrapper(HT, 
                             P_HT_rcs, 
                             e, 
                             eT, 
                             C);

    //P_HT_rcs = sparse_lil_2_rcs(ab->P_HT);
    sparse_rcs_export(P_HT_fname, P_HT_rcs);

    sparse_rcs_destroy(&H);
    sparse_rcs_destroy(&HT);
    sparse_rcs_destroy(&P_HT_rcs);
    ensemble_destroy(&/*ab->*/e);
    sb_toe_r_destroy(&/*ab->*/C);
    //sb_toe_r_nz_it_destroy(&ab->C_it);
    //sparse_lil_destroy(&ab->P_HT);
    //free(ab->config);
    //free(ab);
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "Usage %s <P_HT_fname> <e_fname> <H_fname> <C_fname>\n", argv[0]);
        return 1;
    }

    do_compute_P_HT(argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);

    return 0;
}
