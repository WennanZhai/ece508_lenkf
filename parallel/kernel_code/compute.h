#ifndef COMPUTE_H
#define COMPUTE_H 

void critical_kernel_wrapper(sparse_rcs * HT, 
                             sparse_rcs * P_HT_rcs, 
                             ensemble * e, 
                             ensemble * eT, 
                             sparse_coo * C);

sparse_rcs * cudaMalloc_sparse_rcs(int N, int m, int n);
int cudaMemcpy_sparse_rcs(sparse_rcs * dev, sparse_rcs * host, int hostToDevice);
int cudaFree_sparse_rcs(sparse_rcs * dev);
full_r * cudaMalloc_full_r(int m, int n);
int cudaMemcpy_full_r(full_r * dev, full_r * host, int hostToDevice);
int cudaFree_full_r(full_r * dev);
sparse_coo * cudaMalloc_sparse_coo(int m, int n, int N);
int cudaMemcpy_sparse_coo(sparse_coo * dev, sparse_coo * host, int hostToDevice);
int cudaFree_sparse_coo(sparse_coo * dev);

#endif