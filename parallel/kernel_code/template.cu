#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 512
#define TILE_SIZE 1024

// Ceiling funciton for X / Y.
__host__ __device__ static inline int ceil_div(int x, int y) {
    return (x - 1) / y + 1;
}
/******************************************************************************
 GPU kernels
*******************************************************************************/

/*
 * Sequential merge implementation is given. You can use it in your kernels.
 */
__device__ void merge_sequential(float* A, int A_len, float* B, int B_len, float* C) {
    int i = 0, j = 0, k = 0;

    while ((i < A_len) && (j < B_len)) {
        C[k++] = A[i] <= B[j] ? A[i++] : B[j++];
    }

    if (i == A_len) {
        while (j < B_len) {
            C[k++] = B[j++];
        }
    } else {
        while (i < A_len) {
            C[k++] = A[i++];
        }
    }
}

__device__ int co_rank(int k, float* A, int m, float* B, int n){
    int low = (k > n) ? k-n : 0;
    int high = (k < m) ? k : m;
    while (low < high) {
        int i = low + (high-low) / 2;
        int j = k-i;
        if(i>0 && j<n && A[i-1] > B[j]) {
            high = i-1;
        } else if(j>0 && i<m && A[i] <= B[j-1]) {
            low = i+1;
        } else {
            return i;
        }
    }
    return low;
}

/*
 * Basic parallel merge kernel using co-rank function
 * A, A_len - input array A and its length
 * B, B_len - input array B and its length
 * C - output array holding the merged elements.
 *      Length of C is A_len + B_len (size pre-allocated for you)
 */
__global__ void gpu_merge_basic_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    /* Your code here */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tt = blockDim.x * gridDim.x;
    int m = A_len;
    int n = B_len;
    int elt = ceil_div(m+n, tt);
    int k_curr = min(tid*elt, m+n);
    int k_next = min(k_curr+elt, m+n);

    int i_curr = co_rank(k_curr, A, m, B, n);
    int j_curr = k_curr-i_curr;
    int i_next = co_rank(k_next, A, m, B, n);
    int j_next = k_next-i_next;
    merge_sequential(&A[i_curr], i_next-i_curr, &B[j_curr], j_next-j_curr, &C[k_curr]);
}

/*
 * Arguments are the same as gpu_merge_basic_kernel.
 * In this kernel, use shared memory to increase the reuse.
 */
__global__ void gpu_merge_tiled_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    /* Your code here */
    extern __shared__ float sharedABC[];
    float* tileA = &sharedABC[0];
    float* tileB = &sharedABC[TILE_SIZE];
    float* tileC = &sharedABC[2*TILE_SIZE];

    int m = A_len;
    int n = B_len;

    int elb = ceil_div(m+n, gridDim.x);
    int blk_C_sta = blockIdx.x * elb;
    int blk_C_end = min(blk_C_sta+elb, m+n);

    if(threadIdx.x == 0){
        tileA[0] = co_rank(blk_C_sta, A, m, B, n);
        tileA[1] = co_rank(blk_C_end, A, m, B, n);
    }
    __syncthreads();

    int blk_A_sta = tileA[0];
    int blk_A_end = tileA[1];
    int blk_B_sta = blk_C_sta - blk_A_sta;
    int blk_B_end = blk_C_end - blk_A_end;
    __syncthreads();

    int C_length = blk_C_end - blk_C_sta;
    int A_length = blk_A_end - blk_A_sta;
    int B_length = blk_B_end - blk_B_sta;

    int num_tiles = ceil_div(C_length, TILE_SIZE);
    int C_produced = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    for(int cnt = 0; cnt < num_tiles; cnt++){
        int C_remaining = C_length - C_produced;
        int A_remaining = A_length - A_consumed;
        int B_remaining = B_length - B_consumed;

        for(int i = 0; i < TILE_SIZE; i += blockDim.x){
            int idx = i + threadIdx.x;
            if(idx < A_remaining){
                tileA[idx] = A[blk_A_sta + A_consumed + idx]; 
            }
            if(idx < B_remaining){
                tileB[idx] = B[blk_B_sta + B_consumed + idx]; 
            }
        }
        __syncthreads();

        int elt = ceil_div(TILE_SIZE, blockDim.x);

        int thr_C_curr = min(C_remaining, threadIdx.x * elt);
        int thr_C_next = min(C_remaining, thr_C_curr + elt);

        int C_in_tile = min(TILE_SIZE, C_remaining);
        int A_in_tile = min(TILE_SIZE, A_remaining);
        int B_in_tile = min(TILE_SIZE, B_remaining);

        int thr_A_curr = co_rank(thr_C_curr, tileA, A_in_tile, tileB, B_in_tile);
        int thr_A_next = co_rank(thr_C_next, tileA, A_in_tile, tileB, B_in_tile);
        int thr_B_curr = thr_C_curr - thr_A_curr;
        int thr_B_next = thr_C_next - thr_A_next;

        merge_sequential(&tileA[thr_A_curr], thr_A_next-thr_A_curr,
                         &tileB[thr_B_curr], thr_B_next-thr_B_curr, &tileC[thr_C_curr]);
        __syncthreads();

        for(int j=0; j < TILE_SIZE; j+= blockDim.x){
            int idx = j + threadIdx.x;
            if(idx < C_in_tile){
                C[blk_C_sta+C_produced+idx] = tileC[idx];
            }
        }
        __syncthreads();

        C_produced += C_in_tile;
        A_consumed += co_rank(C_in_tile, tileA, A_in_tile, tileB, B_in_tile);
        B_consumed = C_produced - A_consumed;
        __syncthreads();
    }
    
}

/*
 * gpu_merge_circular_buffer_kernel is optional.
 * The implementation will be similar to tiled merge kernel.
 * You'll have to modify co-rank function and sequential_merge
 * to accommodate circular buffer.
 */
__global__ void gpu_merge_circular_buffer_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    /* Your code here */
}

/******************************************************************************
 Functions
*******************************************************************************/

void gpu_basic_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    gpu_merge_basic_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}

void gpu_tiled_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    int tile_size = 3*TILE_SIZE*sizeof(float);
    gpu_merge_tiled_kernel<<<numBlocks, BLOCK_SIZE, tile_size>>>(A, A_len, B, B_len, C);
}

void gpu_circular_buffer_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    gpu_merge_circular_buffer_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}
