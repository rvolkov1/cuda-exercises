#include <iostream>
#include <cuda_runtime.h>

__global__ void vector_add_series(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) {
      c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_parallel(float *a, float *b, float *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

int main() {
    const int N = 1000000;

    float *a, *b, *c;
    float *dev_a, *dev_b, *dev_c;

    a = (float*)malloc(sizeof(float) * N);
    b = (float*)malloc(sizeof(float) * N);
    c = (float*)malloc(sizeof(float) * N);

    for (int i = 0; i < N; ++i) {
      a[i] = 1.0f;
      b[i] = 2.0f;
    }

    checkCudaError(cudaMalloc((void**)&dev_a, N * sizeof(float)), "Alloc dev_a");
    checkCudaError(cudaMalloc((void**)&dev_b, N * sizeof(float)), "Alloc dev_b");
    checkCudaError(cudaMalloc((void**)&dev_c, N * sizeof(float)), "Alloc dev_c");

    checkCudaError(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice), "Memcpy a");
    checkCudaError(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice), "Memcpy b");

    int num_threads = 100;

    vector_add_series<<<1, num_threads>>>(dev_a, dev_b, dev_c, N);
    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution");

    checkCudaError(cudaMemcpy(c, dev_c, N * sizeof(float), cudaMemcpyDeviceToHost), "Memcpy c");

    //for (int i = 0; i < size; i++) {
    //    std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    //}

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}