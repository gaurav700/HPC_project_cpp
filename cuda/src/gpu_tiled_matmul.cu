// gpu_tiled_matmul.cu
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>

#define TILE 32

inline void checkCudaErr(cudaError_t e, const char* s){
    if(e != cudaSuccess){ fprintf(stderr, "CUDA err %s: %s\n", s, cudaGetErrorString(e)); exit(1); }
}

__global__ void matmul_tile(const float* A, const float* B, float* C, int N){
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    int numTiles = (N + TILE - 1)/TILE;
    for(int t=0; t < numTiles; ++t){
        int aCol = t*TILE + threadIdx.x;
        int bRow = t*TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row*N + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow*N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for(int k=0;k<TILE;++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if(row < N && col < N) C[row*N + col] = acc;
}

int main(int argc,char** argv){
    if(argc < 2){ printf("Usage: %s N\n", argv[0]); return 1; }
    int N = atoi(argv[1]);
    if(N <= 0) return 1;
    size_t elems = (size_t)N * N;

    // deterministic seed
    srand48(12345);
    std::vector<float> A(elems), B(elems), C(elems,0.0f);
    for(size_t i=0;i<elems;++i){ A[i] = (float)drand48(); B[i] = (float)drand48(); }

    float *dA=nullptr, *dB=nullptr, *dC=nullptr;
    checkCudaErr(cudaMalloc((void**)&dA, elems*sizeof(float)), "alloc dA");
    checkCudaErr(cudaMalloc((void**)&dB, elems*sizeof(float)), "alloc dB");
    checkCudaErr(cudaMalloc((void**)&dC, elems*sizeof(float)), "alloc dC");

    checkCudaErr(cudaMemcpy(dA, A.data(), elems*sizeof(float), cudaMemcpyHostToDevice), "H2D A");
    checkCudaErr(cudaMemcpy(dB, B.data(), elems*sizeof(float), cudaMemcpyHostToDevice), "H2D B");

    dim3 block(TILE, TILE);
    dim3 grid((N+TILE-1)/TILE, (N+TILE-1)/TILE);

    cudaEvent_t s,e; checkCudaErr(cudaEventCreate(&s), "evt create s"); checkCudaErr(cudaEventCreate(&e), "evt create e");
    checkCudaErr(cudaEventRecord(s), "evt record s");
    matmul_tile<<<grid, block>>>(dA, dB, dC, N);
    checkCudaErr(cudaGetLastError(), "kernel launch");
    checkCudaErr(cudaEventRecord(e), "evt record e");
    checkCudaErr(cudaEventSynchronize(e), "evt sync e");

    float ms=0.0f; checkCudaErr(cudaEventElapsedTime(&ms, s, e), "elapsed");
    checkCudaErr(cudaMemcpy(C.data(), dC, elems*sizeof(float), cudaMemcpyDeviceToHost), "D2H C");

    double csum = 0.0;
    for(size_t i=0;i<elems;++i) csum += (double)C[i];

    printf("n=%d kernel_ms=%.3f checksum=%.6e\n", N, ms, csum);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return 0;
}
