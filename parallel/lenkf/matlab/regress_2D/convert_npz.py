#!/usr/local/bin/python3

import sys

import scipy.sparse

import libmdb_matrix


if __name__ == '__main__':
    H_output_fname = sys.argv[1]
    H_input_fname = sys.argv[2]

    H_ccs = scipy.sparse.load_npz(H_input_fname)
    H_csr = H_ccs.tocsr()

    libmdb_matrix.export_rcs(H_output_fname, H_csr)
