from numpy import *
from scipy import *
from scipy.fftpack import *

import os.path
import sys
import subprocess

sys.path.append(os.path.expanduser("~/src/radon/trunk"))
import isotropic_filter
from fftmisc import zero_pad, circshift
import convmtx

sys.path.append(os.path.expanduser("~/src/libmdb_matrix"))
import libmdb_matrix


if __name__ == "__main__":
    n = array([3, 3, 6], dtype="int32")
    m = array([2, 6, 9], dtype="int32")

    L = 16

    setup_filename = "/tmp/setup"
    filter_filename = "/tmp/filter"
    output_filename = "/tmp/output"
    randn_filename = "/tmp/randn"
    
    k = m + n - 1

    assert(len(n) == len(m))
    ndim = len(n)

    a1 = array([ndim], dtype="int32")
    a2 = array([L], dtype="int32")

    with open(setup_filename, "w") as fid:
        a1.tofile(fid)
        n.tofile(fid)
        a2.tofile(fid)
    
    norm_m = linalg.norm(m)
    sqrt_shape_function = lambda r: 1 - float(r)/(norm_m)
    H = isotropic_filter.ShapeFilter(n, sqrt_shape_function, m)

    flip = [slice(None, None, -1) for n_i in n]
    h_sqrt_zp = circshift(zero_pad(H.h[flip], k), -(m-1))

    libmdb_matrix.export_r_filter_new(filter_filename, h_sqrt_zp, dtype="float32")

    w = asarray(randn(prod(k), L), dtype="double")
    with open("/tmp/randn", "w") as fid:
        for l in range(L):
            z = w[:, l]
            z.shape = k
            z.tofile(fid)

    Q_sqrt = H.asmatrix().T

    e_python = Q_sqrt * w

    subprocess.check_call(["../randn_conv_test_new", setup_filename, filter_filename, output_filename])
    e_c = libmdb_matrix.import_full_r(output_filename)

    print(linalg.norm((e_python - e_c).flat))
