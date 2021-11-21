# libmdb_matrix

A C-based matrix library. Full, full symmetric (triangular storage), sparse, and Toeplitz (i.e., filters) are supported.

## Dependencies

This code has been known to build on Linux systems and macOS (Catalina).

The following libraries are required on all systems:
- GNU Scientific Library (GSL)
- Fastest Fourier Transform in the West (FFTW)

In addition, the following are required on Linux systems:
- BLAS routines (specifically, OpenBLAS)
- LAPACKE (C interface to the LAPACK routines)

## Compilation

The library should build with an execution of `make`. A successful build will produce:
- `libmdb_matrix_d.so` (double precision matrix library)
- `libmdb_matrix_s.so` (single precision matrix library)
- `*_test` a suite of test case, standalone programs. The expected output is documented in each test case program.

## Interfaces

The `matlab` and `python` subdirectories contain Matlab and Python code for importing and exporting the more commonly used matrix and vector formats supported by `libmdb_matrix`. The Python module includes a `setup.py` file for installing the module (you can `python3 setup.py develop` and then `import libmdb_matrix` in Python to verify a successful installation).
