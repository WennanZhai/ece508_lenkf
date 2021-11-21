import numpy as np
import scipy as sp
import scipy.sparse


def map_bytes_to_dtype(nbytes, int_type=False):
    # Should use a dict instead
    if (nbytes == 4) and (not int_type):
        return 'float32'
    elif (nbytes == 8) and (not int_type):
        return 'float64'
    elif (nbytes == 4) and int_type:
        return 'int32'
    elif (nbytes == 8) and int_type:
        return 'int64'
    else:
        raise ValueError('nbytes must be 4 or 8 but {0} passed to function'.format(nbytes))


def import_full_c(filename):
    with open(filename, 'r') as fid:
        nbytes = int(np.fromfile(fid, dtype=np.int32, count=1))
        m = int(np.fromfile(fid, dtype=np.int32, count=1))
        n = int(np.fromfile(fid, dtype=np.int32, count=1))

        x = reshape(np.fromfile(fid,
                                dtype=map_bytes_to_dtype(nbytes)),
                    (m, n),
                    order='F')

        return x


def import_full_r(filename):
    with open(filename, 'r') as fid:
        nbytes = int(np.fromfile(fid, dtype=np.int32, count=1))
        m = int(np.fromfile(fid, dtype=np.int32, count=1))
        n = int(np.fromfile(fid, dtype=np.int32, count=1))

        x = np.fromfile(fid, dtype=map_bytes_to_dtype(nbytes))
        x.shape = (m, n)
        return x


def import_vector(filename):
    with open(filename, 'r') as fid:
        nbytes = int(np.fromfile(fid, dtype=np.int32, count=1))
        n = int(np.fromfile(fid, dtype=np.int32, count=1))

        x = np.fromfile(fid, dtype=map_bytes_to_dtype(nbytes))
        assert(len(x) == n)
        return x


def import_ivector(filename):
    with open(filename, 'r') as fid:
        nbytes = int(np.fromfile(fid, dtype=np.int32, count=1))
        n = int(np.fromfile(fid, dtype=np.int32, count=1))

        x = np.fromfile(fid, dtype=map_bytes_to_dtype(nbytes, int_type=True))
        assert(len(x) == n)
        return x


def import_rcs(filename):
    with open(filename, 'r') as fid:
        nbytes = int(np.fromfile(fid, dtype=np.int32, count=1))
        m = int(np.fromfile(fid, dtype=np.int32, count=1))
        n = int(np.fromfile(fid, dtype=np.int32, count=1))
        N = int(np.fromfile(fid, dtype=np.int32, count=1))

        v = np.fromfile(fid, dtype=map_bytes_to_dtype(nbytes), count=N)
        j = np.fromfile(fid, dtype=np.int32, count=N)
        r = np.fromfile(fid, dtype=np.int32, count=m+1)

        return sp.sparse.csr_matrix((v, j, r), shape=(m, n))


def import_diag(filename):
    with open(filename, 'r') as fid:
        nbytes = int(np.fromfile(fid, dtype=np.int32, count=1))
        m = int(np.fromfile(fid, dtype=np.int32, count=1))
        n = int(np.fromfile(fid, dtype=np.int32, count=1))
        v = np.fromfile(fid, dtype=map_bytes_to_dtype(nbytes), count=min(m,n))

        return sp.sparse.spdiags(v, 0, m, n)


def import_sb_toe_r(filename, verbose=False):
    """Load a recursive symmetric block Toeplitz matrix and store it in
    COO format. This an inefficient way to represent the matrix --- it
    would be better to use, e.g., a block sparse row representation,
    but this is also not a perfect fit as we would want to store it as
    a recursive block sparse row matrix (i.e., sparse blocks of sparse
    blocks of ... of symmetric Toeplitz).
    """
    with open(filename, 'r') as fid:
        nbytes = int(np.fromfile(fid, dtype=np.int32, count=1))
        rank = int(np.fromfile(fid, dtype=np.int32, count=1))
        assert rank > 0
        n_phy = np.fromfile(fid, dtype=np.int32, count=rank)
        n = np.fromfile(fid, dtype=np.int32, count=rank)
        v = np.fromfile(fid, dtype=map_bytes_to_dtype(nbytes))
        assert len(v) == np.prod(n_phy)
    if verbose:
        print(f'rank = {rank}')
        print(f'n = {n}')
        print(f'n_phy = {n_phy}')
    # now construct coo matrix
    N = np.prod(n)
    A_v = []
    A_i = []
    A_j = []
    block_size = np.cumprod(n[::-1])[::-1][1:]
    phy_block_size = np.append(np.cumprod(n_phy[::-1])[::-1][1:], 1)
    for row in range(N):
        for col in range(N):
            b_i = row
            b_j = col
            b = []
            for block_size_k in block_size:
                r1, b_i = divmod(b_i, block_size_k)
                r2, b_j = divmod(b_j, block_size_k)
                b.append(abs(r1 - r2))
            b.append(abs(b_i - b_j))
            if not any(b >= n_phy):
                index_v = np.sum(b * phy_block_size)
                A_v.append(v[index_v])
                A_i.append(row)
                A_j.append(col)
    return scipy.sparse.coo_matrix((A_v, (A_i, A_j)), shape=(N, N))


def import_sparse_coo(fname, dtype=np.double):
    with open(fname, 'rb') as fid:
        sizeof_elem = np.fromfile(fid, dtype=np.int32, count=1)[0]
        assert sizeof_elem == np.dtype(dtype).itemsize
        m = np.fromfile(fid, dtype=np.int32, count=1)[0]
        n = np.fromfile(fid, dtype=np.int32, count=1)[0]
        N = np.fromfile(fid, dtype=np.int32, count=1)[0]

        v = np.fromfile(fid, dtype=dtype, count=N)
        i = np.fromfile(fid, dtype=np.int32, count=N)
        j = np.fromfile(fid, dtype=np.int32, count=N)
    return sp.sparse.coo_matrix((v, (i, j)), shape=[m, n])


################################################################################


def export_full_r(filename, x, dtype=np.double):
    with open(filename, 'w') as fid:
        m, n = x.shape

        z = np.array([np.dtype(dtype).itemsize], dtype='int32')
        z.tofile(fid)

        z = np.array([m, n], dtype='int32')
        z.tofile(fid)

        if (x.dtype != dtype):
            x_dtype = np.array(x, dtype=dtype)
            x_dtype.tofile(fid)
        else:
            x.tofile(fid)


def export_vector(filename, x, dtype=np.double):
    with open(filename, 'w') as fid:
        n = len(x)

        z = np.array([np.dtype(dtype).itemsize], dtype='int32')
        z.tofile(fid)

        z = np.array([n], dtype='int32')
        z.tofile(fid)

        if (x.dtype != dtype):
            x_dtype = np.array(x, dtype=dtype)
            x_dtype.tofile(fid)
        else:
            x.tofile(fid)


def export_ivector(filename, x, dtype=np.int64):
    export_vector(filename, np.asarray(x, dtype=dtype), dtype=dtype)


def export_rcs(filename, A, dtype=np.double):
    with open(filename, 'w') as fid:
        A = sp.sparse.csr_matrix(A, dtype=dtype)

        m, n = A.shape
        N = A.nnz

        v = A.data
        j = A.indices
        r = A.indptr

        z = np.array([np.dtype(dtype).itemsize, m, n, N], dtype='int32')
        z.tofile(fid)

        assert(j.dtype == 'int32')
        assert(r.dtype == 'int32')

        v.tofile(fid)
        j.tofile(fid)
        r.tofile(fid)


def export_diag(filename, A, dtype=np.double):
    with open(filename, 'w') as fid:
        m, n = A.shape
        v = np.asarray(A.diagonal(), dtype=dtype)

        z = np.array([np.dtype(dtype).itemsize, m, n], dtype=int32)
        z.tofile(fid)

        v.tofile(fid)


def export_r_filter_new(filename, H, dtype=np.double):
    with open(filename, 'w') as fid:
        header = np.array([np.dtype(dtype).itemsize, H.ndim], dtype='int32')
        n_log = np.array(H.shape, dtype='int32')
        header.tofile(fid)
        n_log.tofile(fid)

        H_dtype = np.asarray(H, dtype=dtype)
        H_dtype.tofile(fid)


def export_sb_toe_r(filename, blocks, dtype=np.double):
    rank = len(blocks.shape)

    n = blocks.shape

    n_phy = []
    for dim in range(rank):
        index = [slice(None)] * rank
        count = 0
        for i in range(n[dim]):
            index[dim] = slice(i, i+1)
            if any(blocks[tuple(index)].flat):
                count += 1
        n_phy.append(count)

    with open(filename, 'w') as fid:
        z = np.array([np.dtype(dtype).itemsize], dtype='int32')
        z.tofile(fid)

        z = np.array([rank], dtype='int32')
        z.tofile(fid)

        z = np.array(n_phy[::-1], dtype='int32')
        z.tofile(fid)

        z = np.array(n[::-1], dtype='int32')
        z.tofile(fid)

        index = tuple(slice(None, x) for x in n_phy)
        blocks_phy = blocks[index]

        z = np.array(blocks_phy.reshape((np.prod(n_phy),),
                                        order='F'), dtype=dtype)
        z.tofile(fid)

    return filename
