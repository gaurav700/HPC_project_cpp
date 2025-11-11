// gpu_cublas_matmul.cu
// Simple cuBLAS-backed matrix multiplication benchmark (single-precision)
// Measures only the cuBLAS kernel time (H2D/D2H excluded from timing)

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

inline void checkCudaErr(cudaError_t e, const char* s){
    if(e != cudaSuccess){ fprintf(stderr, "CUDA err %s: %s\n", s, cudaGetErrorString(e)); exit(1); }
}
inline void checkCublasErr(cublasStatus_t s, const char* msg){
    if(s != CUBLAS_STATUS_SUCCESS){ fprintf(stderr, "cuBLAS err %s: %d\n", msg, (int)s); exit(1); }
}

int main(int argc,char** argv){
    if(argc < 2){ printf("Usage: %s N\n", argv[0]); return 1; }
    int N = atoi(argv[1]);
    if(N <= 0) return 1;
    size_t elems = (size_t)N * N;

    srand48(12345);
    std::vector<float> A(elems), B(elems), C(elems, 0.0f);
    for(size_t i=0;i<elems;++i){ A[i] = (float)drand48(); B[i] = (float)drand48(); }

    // cuBLAS expects column-major layout. We'll transpose host matrices before copying
    std::vector<float> At(elems), Bt(elems);
    for(int r=0;r<N;++r){
        for(int c=0;c<N;++c){
            At[(size_t)c*N + r] = A[(size_t)r*N + c];
            Bt[(size_t)c*N + r] = B[(size_t)r*N + c];
        }
    }

    float *dA=nullptr, *dB=nullptr, *dC=nullptr;
    checkCudaErr(cudaMalloc((void**)&dA, elems*sizeof(float)), "alloc dA");
    checkCudaErr(cudaMalloc((void**)&dB, elems*sizeof(float)), "alloc dB");
    checkCudaErr(cudaMalloc((void**)&dC, elems*sizeof(float)), "alloc dC");

    // copy transposed (column-major) data
    checkCudaErr(cudaMemcpy(dA, At.data(), elems*sizeof(float), cudaMemcpyHostToDevice), "H2D A");
    checkCudaErr(cudaMemcpy(dB, Bt.data(), elems*sizeof(float), cudaMemcpyHostToDevice), "H2D B");

    cublasHandle_t handle;
    checkCublasErr(cublasCreate(&handle), "create handle");

    const float alpha = 1.0f, beta = 0.0f;
    // C = alpha * A * B + beta * C
    // All matrices are stored in column-major on device

    cudaEvent_t s,e; checkCudaErr(cudaEventCreate(&s), "evt create s"); checkCudaErr(cudaEventCreate(&e), "evt create e");
    checkCudaErr(cudaEventRecord(s), "evt record s");

    // cublasSgemm: column-major multiplication
    // A: N x N, B: N x N, C: N x N
    checkCublasErr(cublasSgemm(handle,
                               CUBLAS_OP_N, CUBLAS_OP_N,
                               N, N, N,
                               &alpha,
                               dA, N,
                               dB, N,
                               &beta,
                               dC, N), "sgemm");

    checkCudaErr(cudaEventRecord(e), "evt record e");
    checkCudaErr(cudaEventSynchronize(e), "evt sync e");

    float ms=0.0f; checkCudaErr(cudaEventElapsedTime(&ms, s, e), "elapsed");

    // copy result back (transposed column-major C)
    checkCudaErr(cudaMemcpy(C.data(), dC, elems*sizeof(float), cudaMemcpyDeviceToHost), "D2H C");

    // transpose back to row-major to compute checksum consistent with tiled implementation
    double csum = 0.0;
    for(int r=0;r<N;++r){
        for(int c=0;c<N;++c){
            // device C is column-major, element (r,c) is at index c*N + r
            float val = C[(size_t)c*N + r];
            csum += (double)val;
        }
    }

    printf("n=%d kernel_ms=%.3f checksum=%.6e\n", N, ms, csum);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cublasDestroy(handle);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return 0;
}
