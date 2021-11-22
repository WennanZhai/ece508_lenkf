#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt

from libmdb_matrix import import_sb_toe_r


if __name__ == '__main__':
    C_fname = sys.argv[1]
    C = import_sb_toe_r(C_fname, verbose=True)

    plt.matshow(C.todense())
    plt.colorbar()
    plt.show()
