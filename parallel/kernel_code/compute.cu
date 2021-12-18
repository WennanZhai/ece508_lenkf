#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <cusparse.h>
#include "helper.hpp"
#include "structs.hu"

#define TILE_SZ_A 128
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A/TILE_SZ_B)
#define BLOCK_SIZE 512

extern "C"{
    #include "compute.h"
}

__global__ void mysgemm(int m, int n, int k, const double *A, const double *B, double* C) {

  /********************************************************************
  *
  * Compute C = A x B
  *   where A is a (m x k) matrix
  *   where B is a (k x n) matrix
  *   where C is a (m x n) matrix
  *
  * Use register and shared memory tiling and thread coarsening
  *
  * NOTE: A and C are column major, B is row major
  *
  ********************************************************************/

  // Macros for accessing flattened matrices
  #define A(row,col) A[(row) + (col)*m]
  #define B(row,col) B[(row)*n + (col)]
  #define C(row,col) C[(row) + (col)*m]

  __shared__ double Bs[TILE_SZ_RATIO][TILE_SZ_B];

  double out[TILE_SZ_B] = {0.0};
  double Ar = 0.0;

  int tx = threadIdx.x;
  int row = blockIdx.x * TILE_SZ_A + tx;
  int col = blockIdx.y * TILE_SZ_B;

  for(int i = 0; i < k; i += TILE_SZ_RATIO)
  {
    Bs[tx / TILE_SZ_B][tx % TILE_SZ_B] = (i + tx / TILE_SZ_B < k && col + tx % TILE_SZ_B < n) ? B(i + tx / TILE_SZ_B, col + tx % TILE_SZ_B) : 0.0;
    __syncthreads();

    for(int j = 0; j < TILE_SZ_RATIO; j++)
    {
      if(row < m && i + j < k)
      {
        Ar = A(row, i + j);
        for(int l = 0; l < TILE_SZ_B; l++)
        {
          out[l] += Ar * Bs[j][l];
        }
      }
    }
    __syncthreads();
  }

  if(row < m)
  {
    for(int i = 0; i < TILE_SZ_B; i++)
    {
      if(col + i < n)
      {
        C(row, col + i) = out[i];
      }
    }
  }
  // SSL Hint (9/6/21): try using just one register for the tile of A 
  // rather than several--in other words, load one value (per thread) 
  // from A and compute using that value rather than loading all values 
  // before doing the computation.  This approach seems to be slightly 
  // faster than the alternative.
  #undef A
  #undef B
  #undef C
}

__global__ void coo_dense_elem_mul(int N, double * v, int * i, int * j, 
                                   double * vector, int m, int n, 
                                   double * v_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N)
    {
        int i_curr = i[idx];
        int j_curr = j[idx];
        if(i_curr < m && j_curr < n)
        {
            v_out[idx] = v[idx] * vector[i_curr * n + j_curr];
        }
    }
}

extern "C" 
void critical_kernel_wrapper(sparse_rcs * HT, 
                             sparse_rcs * P_HT_rcs, 
                             ensemble * e, 
                             ensemble * eT, 
                             sparse_coo * C) {
    
    double *ev, *eTv, *eeTv, *Cv, *CeeTv;
    int *Ci, *Cj;

    int *rcsCi;

    cusparseStatus_t status;
    cusparseHandle_t handle;

    status = cusparseCreate(&handle);

    timer_start("Allocating GPU memory.");
    cudaMalloc((void**) &ev, sizeof(double) * (e->X->m * e->X->n));
    cudaMalloc((void**) &eTv, sizeof(double) * (eT->X->m * eT->X->n));
    cudaMalloc((void**) &eeTv, sizeof(double) * (e->X->m * eT->X->n));
    cudaMalloc((void**) &Ci, sizeof(int) * C->N);
    cudaMalloc((void**) &Cj, sizeof(int) * C->N);
    cudaMalloc((void**) &Cv, sizeof(double) * C->N);
    cudaMalloc((void**) &CeeTv, sizeof(double) * C->N);
    cudaMalloc((void**) &rcsCi, sizeof(int) * (C->N + 1));
    timer_stop();

    timer_start("Copying input memory to the GPU.");
    cudaMemcpy(ev, e->X->v_vector, sizeof(double) * (e->X->m * e->X->n), cudaMemcpyHostToDevice);
    cudaMemcpy(eTv, eT->X->v_vector, sizeof(double) * (e->X->m * e->X->n), cudaMemcpyHostToDevice);
    cudaMemcpy(Ci, C->i, sizeof(int) * C->N, cudaMemcpyHostToDevice);
    cudaMemcpy(Cj, C->j, sizeof(int) * C->N, cudaMemcpyHostToDevice);
    cudaMemcpy(Cv, C->v, sizeof(double) * C->N, cudaMemcpyHostToDevice);
    timer_stop();

    dim3 dimBlockSgemm(TILE_SZ_A, 1, 1);
    dim3 dimGridSgemm(ceil(e->X->m * 1.0 / TILE_SZ_A), ceil(eT->X->n * 1.0 / TILE_SZ_B), 1);

    dim3 dimBlockCooDense(BLOCK_SIZE, 1, 1);
    dim3 dimGridCooDense(ceil(C->N / (1.0 * BLOCK_SIZE)), 1, 1);

    timer_start("Performing GPU Critical Step computation");
    //Here we multiply e against eT giving us the values in vector eeTv
    mysgemm<<<dimGridSgemm, dimBlockSgemm>>>(e->X->m, eT->X->n, e->X->n, ev, eTv, eeTv);
    cudaDeviceSynchronize();
    //Next we multiply the dense eeTv vector against the coo sparse C data
    coo_dense_elem_mul<<<dimGridCooDense, dimBlockCooDense>>>(C->N, Cv, Ci, Cj,
                                                                eeTv, e->X->m, eT->X->n,
                                                                CeeTv);
    cudaDeviceSynchronize();
    //The output coo sparse from the previous step is converted to csr
    //Csr simply has a compressed row vector, but is the same as coo otherwise
        //thus Ci is converted to rcsCi
    status = cusparseXcoo2csr(handle, Ci, C->N, C->m, rcsCi, CUSPARSE_INDEX_BASE_ZERO);
    cudaDeviceSynchronize();
    //Sparse Matrix Multiplication of 2 csr sparse matrices (rcsCi, Cj, CeeTv) & (HT->r, HT->j, HT->v)
    timer_stop();

    timer_start("Copying output memory to the CPU");
    timer_stop();

    cudaFree(ev);
    cudaFree(eTv);
    cudaFree(eeTv);

    fprintf(stderr, "KERNEL RAN SUCCESFULLY\n");
}

// //Create space for sparse rcs on device and return pointer
// sparse_rcs * cudaMalloc_sparse_rcs(int N, int m, int n) {
//   sparse_rcs *A;

//   //arg bound check
//   assert(m >= 0);
//   assert(n >= 0);
//   assert(N >= 0);

//   int dim[3] = {N, m, n};
//   fprintf(stderr, "here0\n");
//   //create struct itself
//   cudaMalloc((void**) &A, sizeof(sparse_rcs));
//   fprintf(stderr, "here1\n");

//   //copy dimensions into struct
//   cudaMemcpy(A, dim, sizeof(int) * 3, cudaMemcpyHostToDevice);
//   fprintf(stderr, "here2\n");

//   //copy all associated data
//   if (m > 0) {
//     int * Am;
//     fprintf(stderr, "here5\n");
//     cudaMalloc((void **) &(A->r), sizeof(int) * (m+1));
//     fprintf(stderr, "here6\n");
//   }
//   else {
//     //A->r = NULL;
//   }
//   fprintf(stderr, "here4\n");

//   if (N == 0) {
//     //A->v = NULL;
//     //A->j = NULL;
//   }
//   else {
//     cudaMalloc((void **) &(A->v), sizeof(double) * N);
//     //cudaMemcpy(A->v, B->v, sizeof(double) * N, cudaMemcpyHostToDevice);

//     cudaMalloc((void **) &(A->j), sizeof(double) * N);
//     //cudaMemcpy(A->j, B->j, sizeof(double) * N, cudaMemcpyHostToDevice);
//   }
//   fprintf(stderr, "here3\n");

//   return A;
// }

// //copy data from host sparse rcs to device rcs if dims match
// int cudaMemcpy_sparse_rcs(sparse_rcs * dev, sparse_rcs * host, int hostToDevice)
// {
//     if(hostToDevice == 1)
//     {
//         cudaMemcpy(dev->r, host->r, sizeof(int) * (host->m+1), cudaMemcpyHostToDevice);
//         cudaMemcpy(dev->j, host->j, sizeof(double) * host->N, cudaMemcpyHostToDevice);
//         cudaMemcpy(dev->v, host->v, sizeof(double) * host->N, cudaMemcpyHostToDevice);
//         return 0;
//     }
//     else if(hostToDevice == 0)
//     {
//         cudaMemcpy(host->r, dev->r, sizeof(int) * (host->m+1), cudaMemcpyDeviceToHost);
//         cudaMemcpy(host->j, dev->j, sizeof(double) * host->N, cudaMemcpyDeviceToHost);
//         cudaMemcpy(host->v, dev->v, sizeof(double) * host->N, cudaMemcpyDeviceToHost);
//         return 0;
//     }
//     return -1;
// }

// int cudaFree_sparse_rcs(sparse_rcs * dev)
// {
//     cudaFree(dev->r);
//     cudaFree(dev->j);
//     cudaFree(dev->v);
//     cudaFree(dev);
//     return 0;
// }

// full_r * cudaMalloc_full_r(int m, int n)
// {
//     full_r * A;

//     //arg bound check
//     assert(m >= 0);
//     assert(n >= 0);

//     int dims[2] = {m, n};

//     cudaMalloc((void**) &(A), sizeof(full_r));
//     cudaMemcpy(A, dims, sizeof(int) * 2, cudaMemcpyHostToDevice);

//     cudaMalloc((void**) &(A->v_vector), m * n * sizeof(double));

//     return A;
// }

// int cudaMemcpy_full_r(full_r * dev, full_r * host, int hostToDevice)
// {
//     if(hostToDevice == 1)
//     {
//         cudaMemcpy(dev->v_vector, host->v_vector, host->m * host->n * sizeof(double), cudaMemcpyHostToDevice);
//         return 0;
//     }
//     else if(hostToDevice == 0)
//     {
//         cudaMemcpy(host->v_vector, dev->v_vector, host->m * host->n * sizeof(double), cudaMemcpyDeviceToHost);
//         return 0;
//     }
//     return -1;
// }

// int cudaFree_full_r(full_r * dev)
// {
//     cudaFree(dev->v_vector);
//     cudaFree(dev);
//     return 0;
// }

// sparse_coo * cudaMalloc_sparse_coo(int m, int n, int N)
// {
//     sparse_coo * A;

//     assert(m >= 0);
//     assert(n >= 0);
//     assert(N >= 0);

//     int dims[3] = {m, n, N};

//     cudaMalloc((void**) &A, sizeof(sparse_coo));
//     cudaMemcpy(A, dims, 3 * sizeof(int), cudaMemcpyHostToDevice);

//     cudaMalloc((void**) &(A->v), sizeof(double) * N);
//     cudaMalloc((void**) &(A->i), sizeof(int) * N);
//     cudaMalloc((void**) &(A->j), sizeof(int) * N);

//     return A;
// }

// int cudaMemcpy_sparse_coo(sparse_coo * dev, sparse_coo * host, int hostToDevice)
// {
//     if(hostToDevice == 1)
//     {
//         cudaMemcpy(dev->v, host->v, dev->N * sizeof(double), cudaMemcpyHostToDevice);
//         cudaMemcpy(dev->i, host->i, dev->N * sizeof(int), cudaMemcpyHostToDevice);
//         cudaMemcpy(dev->j, host->j, dev->N * sizeof(int), cudaMemcpyHostToDevice);
//         return 0;
//     }
//     else if(hostToDevice == 0)
//     {
//         cudaMemcpy(host->v, dev->v, dev->N * sizeof(double), cudaMemcpyDeviceToHost);
//         cudaMemcpy(host->i, dev->i, dev->N * sizeof(int), cudaMemcpyDeviceToHost);
//         cudaMemcpy(host->j, dev->j, dev->N * sizeof(int), cudaMemcpyDeviceToHost);
//         return 0;
//     }
//     return -1;
// }

// int cudaFree_sparse_coo(sparse_coo * dev)
// {
//     cudaFree(dev->v);
//     cudaFree(dev->j);
//     cudaFree(dev->i);
//     cudaFree(dev);
//     return 0;
// }
