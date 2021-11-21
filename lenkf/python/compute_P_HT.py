#!/usr/bin/env python3

import sys

import sh
import numpy as np
import scipy.sparse
from scipy.linalg import toeplitz

from C_matrix import C_matrix
from libmdb_matrix import export_full_r, export_rcs, export_sb_toe_r, import_rcs, import_sb_toe_r


if __name__ == '__main__':
    np.random.seed(0)

    # Example with 1 physical dimension
    N = 100
    L = 10
    M = 20

    H_density = 0.1

    P_HT_fname = '/tmp/P_HT'
    e_fname = '/tmp/e'
    H_fname = '/tmp/H'
    C_fname = '/tmp/C'

    e = np.random.randn(N, L)
    export_full_r(e_fname, e)

    H = scipy.sparse.random(M, N, density=H_density, format='csr')
    export_rcs(H_fname, H)

    assert N % 2 == 0
    N2 = int(N/2)
    row0 = np.hstack((np.arange(N2, 0, -1), np.zeros((N2,))))
    blocks = row0
    export_sb_toe_r(C_fname, blocks)

    C = toeplitz(row0)

    # RUN CODE
    compute_P_HT_c = sh.Command('../compute_P_HT')
    print(compute_P_HT_c(P_HT_fname,
                         e_fname,
                         H_fname,
                         C_fname,
                         _err_to_out=True,
                         _out=sys.stdout))

    P_HT_c = import_rcs(P_HT_fname)

    P_HT = np.multiply(C, e @ e.T) @ H.T / (L - 1)

    print('P_HT_c == P_HT? {}'.format(np.allclose(P_HT_c.toarray(),
                                                  P_HT)))
    print()


    # Example with 2 physical dimensions
    n = 10, 10
    N = np.prod(n)
    n2 = 3, 3  # controls sparsity of C
    L = 10
    M = 20

    H_density = 0.1

    P_HT_fname = '/tmp/P_HT_2D'
    e_fname = '/tmp/e_2D'
    H_fname = '/tmp/H_2D'
    C_fname = '/tmp/C_2D'

    e = np.random.randn(N, L)
    export_full_r(e_fname, e)

    H = scipy.sparse.random(M, N, density=H_density, format='csr')
    export_rcs(H_fname, H)

    C_iso = C_matrix(n, n2)
    blocks = np.reshape(C_iso.asmatrix()[0, :].todense().flat,
                        n, order='F')

    export_sb_toe_r(C_fname, blocks)

    # RUN CODE
    compute_P_HT_c = sh.Command('../compute_P_HT')
    print(compute_P_HT_c(P_HT_fname,
                         e_fname,
                         H_fname,
                         C_fname,
                         _err_to_out=True,
                         _out=sys.stdout))

    P_HT_c = import_rcs(P_HT_fname)

    # There are some filter edge effects that exist between the
    # isotropic filter presentation of C (note that C_iso.asmatrix()
    # is not a recursive, symmetric block Toeplitz matrix) and the
    # recursive, symmetric block Toeplitz representation. I do not
    # recall the details. Instead, load C as it is stored to disk.

    # C = C_iso.asmatrix().todense()
    C = import_sb_toe_r(C_fname).todense()

    P_HT = np.multiply(C, e @ e.T) @ H.T / (L - 1)

    print('P_HT_c == P_HT? {}'.format(np.allclose(P_HT_c.toarray(),
                                                  P_HT)))
