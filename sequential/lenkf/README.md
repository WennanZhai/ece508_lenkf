# lenkf

A C-based localized ensemble Kalman filter.

## Dependencies

This code has been known to build on Linux systems and macOS (Catalina).

The following libraries are required on all systems:
- `libmdb_matrix`
- GNU Scientific Library (GSL)
- [`libconfuse`](https://github.com/martinh/libconfuse)
- Fastest Fourier Transform in the West (FFTW)

## Compilation

Modify the `PATH_MDB_MATRIX` variable in the `Makefile` to the location of the `libmdb_matrix` code.

The code should build with the execution of `make`. A successful build will produce:
- `lenkf` (engine for running a localized ensemble Kalman filter)
- `compute_P_HT` (driver program for the critical algorithm step)
- Some other test case programs (it's been a while since I have run them).

## Critical step test case generation and validation

The Python code `python/compute_P_HT.py` generates a test case of arbitrary dimension, outputs matrices in formats expected by the C code, calls `compute_P_HT` to compute the critical step result with the C code, computes the same result with a Python one liner, and numerical compares the results.

The problem dimensions are specified at the beginning of the main routine:
- `N`: the state dimension
- `L`: the number of ensemble members (typically `L` << `N`)
- `M`: the number of measurements per time step (typically `M` < `N`)

The Python implementation of the critical step is this line from the main routine:

``` python
P_HT = np.multiply(C, e @ e.T) @ H.T / (L - 1)
```
