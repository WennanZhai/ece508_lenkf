#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt

from libmdb_matrix import import_sparse_coo


if __name__ == '__main__':
    C_fname = sys.argv[1]
    C = import_sparse_coo(C_fname)

    plt.matshow(C.todense())
    plt.colorbar()
    plt.show()
