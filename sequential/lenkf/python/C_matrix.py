#!/usr/bin/env python3

import sys
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np

from isotropic_filter import IsotropicFilter
from libmdb_matrix import export_sb_toe_r


SHAPE_FUN_MAP = {'default': lambda m: lambda r: 1 - r/np.linalg.norm(m)}


def C_matrix(n, m, shape='default'):
    return IsotropicFilter(n,
                           SHAPE_FUN_MAP[shape](m),
                           m,
                           normalized=True)


def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = ArgumentParser('Generate localization matrix.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('C_matrix_fname',
                        type=str,
                        help='file to store the localization matrix')
    parser.add_argument('-n',
                        required=True,
                        nargs='+',
                        type=int,
                        help='grid dimensions')
    parser.add_argument('-m',
                        required=True,
                        nargs='+',
                        type=int,
                        help='square root filter dimensions')
    args = parser.parse_args(argv[1:])

    assert len(args.n) == len(args.m)

    C = C_matrix(args.n,
                 args.m)

    blocks = np.reshape(C.asmatrix()[0, :].todense().flat,
                        args.n,
                        order='F')

    export_sb_toe_r(args.C_matrix_fname, blocks)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
