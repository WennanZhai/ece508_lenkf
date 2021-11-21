#!/usr/bin/env python3

import sh
import matplotlib.pyplot as plt

from libmdb_matrix import import_sparse_coo, import_sb_toe_r


if __name__ == '__main__':
    # First run C code to generate matrices
    toeplitz_new_test = sh.Command('../toeplitz_new_test')
    output = toeplitz_new_test()
    assert output.exit_code == 0

    sb_toe_r_fname = '/tmp/X_toeplitz_new_test'
    X_sb_toe_r = import_sb_toe_r(sb_toe_r_fname)

    coo_matrix_fname = '/tmp/X_coo'
    X_coo = import_sparse_coo(coo_matrix_fname)

    assert (X_sb_toe_r - X_coo).nnz == 0

    plt.matshow(X_sb_toe_r.todense())
    plt.colorbar()

    plt.matshow(X_coo.todense())
    plt.colorbar()

    plt.show()
