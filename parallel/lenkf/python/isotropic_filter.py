import numpy as np
import scipy as sp
import scipy.linalg
import scipy.sparse
import matplotlib.pylab as plt

import libmdb_matrix


class Convmtx(sp.sparse.coo_matrix):
    def __new__(cls, n, H, mode='full'):
        """
        Construct sparse convolution matrix to operate on vector of
        dimension *n* with the kernel *H*. The *mode* parameter can be
        one of:

        - full: standard convolution, i.e., zero-padding at the edges.

        - valid: convolution where only those portions of complete
          overlap, i.e., no zero-padding, are considered.

        - circ: circular convolution, i.e., periodic boundary
          condition at the edges.
        """
        def toeplitz_mapper_full(h):
            if (h == 0).all():
                return sp.sparse.coo_matrix((k[-1], n[-1]))
            else:
                c = h
                r = np.array([c[0]] + [0]*(n[-1]-1))
                return sp.sparse.coo_matrix(scipy.linalg.toeplitz(c, r))

        def toeplitz_mapper_valid(h):
            if (h == 0).all():
                return sp.sparse.coo_matrix((k[-1], n[-1]))
            else:
                r = np.zeros(n[-1])
                r[:len(h)] = h
                c = np.zeros(k[-1])
                c[0] = r[0]
                return sp.sparse.coo_matrix(scipy.linalg.toeplitz(c, r))

        def toeplitz_mapper_circ(h):
            if (h == 0).all():
                return sp.sparse.coo_matrix((k[-1], n[-1]))
            else:
                c = h
                r = np.zeros(n[-1])
                r[0] = c[0]
                r[1:] = h[:0:-1]
                return sp.sparse.coo_matrix(scipy.linalg.toeplitz(c, r))

        def block_mapper_full(n, k, blocks):
            c = [blocks[i] for i in range(k)]
            r = [c[0]] + [None]*(n-1)
            return sp.sparse.bmat(sp.linalg.toeplitz(c, r).tolist(), format='coo')

        def block_mapper_valid(n, k, blocks):
            r = []
            for i in range(n):
                if (n - k - i < 0):
                    r.append(None)
                else:
                    r.append(blocks[n - k - i])

            c = []
            for i in range(n-k, n):
                c.append(blocks[i])

            return sp.sparse.bmat(sp.linalg.toeplitz(c, r).tolist(), format='coo')

        def block_mapper_circ(n, k, blocks):
            c = [blocks[i] for i in range(k)]
            r = []
            r.append(blocks[0])
            r.extend(blocks[:0:-1])
            return sp.sparse.bmat(sp.linalg.toeplitz(c, r).tolist(), format='coo')

        m = H.shape

        if mode == 'full':
            k = tuple(np.array(n) + np.array(m) - 1)
            toeplitz_mapper = toeplitz_mapper_full
            block_mapper = block_mapper_full

            H_zp = zero_pad(H, k)
            c_list = np.split(H_zp.flatten(), np.prod(k[:-1]))
        elif mode == 'valid':
            k = tuple(np.array(n) - np.array(m) + 1)
            toeplitz_mapper = toeplitz_mapper_valid
            block_mapper = block_mapper_valid

            H_zp = zero_pad(H[...,::-1], n)
            c_list = np.split(H_zp.flatten(), np.prod(n[:-1]))
        elif mode == 'circ':
            assert((np.array(m) <= np.array(n)).all())
            k = n
            toeplitz_mapper = toeplitz_mapper_circ
            block_mapper = block_mapper_circ

            H_zp = zero_pad(H, k)
            c_list = np.split(H_zp.flatten(), np.prod(k[:-1]))
        else:
            raise ValueError('Unknown mode {0}'.format(mode))

        blocks = [toeplitz_mapper(x) for x in c_list]

        for n_i, k_i in zip(n[-2::-1], k[-2::-1]):
            if mode == 'full' or mode == 'circ':
                blocks = [block_mapper(n_i, k_i, x) for x in np.split(np.array(blocks), len(blocks)/k_i)]
            elif mode =='valid':
                blocks = [block_mapper(n_i, k_i, x) for x in np.split(np.array(blocks), len(blocks)/n_i)]
            else:
                raise ValueError('Unknown mode {0}'.format(mode))

        return blocks[0]


def zero_pad(x, n):
    """
    Return *x* zero padded to the dimensions specified in *n*.
    """
    assert len(n) == x.ndim
    return np.pad(x,
                  [(0, n_i - s_i) for n_i, s_i in zip(n, x.shape)],
                  'constant')


class ShapeFilter:
    def __init__(self, n, shape_function, m, normalized=False):
        self.n = n
        self.shape_function = shape_function
        self.m = m

        self.h = np.array([shape_function(scipy.linalg.norm(index)) for index in np.ndindex(*m)])
        if normalized:
            self.h /= np.sqrt(sum(self.h * self.h))
        self.h.shape = m


    def asmatrix(self):
        return Convmtx(self.n, self.h)


class IsotropicFilter:
    def __init__(self, n, sqrt_shape_function, m, normalized=False):
        self.n = n
        self.sqrt_shape_function = sqrt_shape_function
        self.m = m

        self.h_sqrt_sf = ShapeFilter(n, sqrt_shape_function, m, normalized=normalized)
        self.h_sqrt = self.h_sqrt_sf.h

    def row0(self):
        s = tuple(slice(None, m_i, None) for m_i in self.m)
        row0 = zero_pad(np.real(np.fft.ifftn(abs(np.fft.fftn(self.h_sqrt,
                                                             s=2*np.array(self.m)-1))**2))[s],
                        self.n)
        row0.shape = (np.prod(self.n),)
        return row0

    def asmatrix(self):
        H_sqrt = self.sqrt_asmatrix()
        return H_sqrt * H_sqrt.T

    def sqrt_asmatrix(self):
        return self.h_sqrt_sf.asmatrix().T

    def h(self, include_zeros=True):
        row0 = self.row0()
        row0.resize(self.n)
        if include_zeros:
            return row0
        else:
            s = [slice(None, m_i, None) for m_i in self.m]
            return row0[s]

    @staticmethod
    def save_sqrt_static(fname, H, dtype='double'):
        H.save_sqrt(fname, dtype=dtype)

    def save_sqrt(self, fname, dtype='double'):
        # Suitable for the sb_toe_r import routines in libmdb_matrix
        with open(fname, 'wb') as fid:
            z = np.array([np.dtype(dtype).itemsize], dtype='int32')
            z.tofile(fid)

            n_dim = np.array([len(self.n)], dtype='int32')
            n_dim.tofile(fid)

            n_phy = np.asarray(self.m, dtype='int32')
            n_phy.tofile(fid)

            n = np.asarray(self.n, dtype='int32')
            n.tofile(fid)

            self.h_sqrt.tofile(fid)


    @staticmethod
    def load_sqrt(fname):
        with open(fname, 'r') as fid:
            itemsize = np.fromfile(fid, dtype='int32', count=1)
            dtype = libmdb_matrix.map_bytes_to_dtype(itemsize)

            n_dim = np.fromfile(fid, dtype='int32', count=1)[0]
            n_phy = np.fromfile(fid, dtype='int32', count=n_dim)
            n = np.fromfile(fid, dtype='int32', count=n_dim)
            h_sqrt = np.fromfile(fid, dtype=dtype, count=np.prod(n_phy))
            h_sqrt.shape = n_phy

            h_sqrt_sf = ShapeFilter.__new__(ShapeFilter)
            h_sqrt_sf.shape_function = None
            h_sqrt_sf.n = n
            h_sqrt_sf.m = n_phy
            h_sqrt_sf.h = h_sqrt

            H = IsotropicFilter.__new__(IsotropicFilter)
            H.n = n
            H.sqrt_shape_function = None
            H.m = n_phy
            H.h_sqrt_sf = h_sqrt_sf
            H.h_sqrt = h_sqrt

            return H


if __name__ == '__main__':
    sqrt_fname = '/tmp/H_sqrt'

    n = (4, 5)

    m = (3, 2)
    norm_m = sp.linalg.norm(m)

    H = IsotropicFilter(n, lambda r: 1 - float(r)/(norm_m), m)
    H.save_sqrt(sqrt_fname)

    H2 = IsotropicFilter.load_sqrt(sqrt_fname)

    print((H.asmatrix() - H2.asmatrix()).nnz)

    plt.matshow(H.h_sqrt)
    plt.colorbar()

    s = tuple(slice(None, m_i, None) for m_i in H.m)

    plt.matshow(H.h()[s])
    plt.colorbar()

    plt.show()
