#include <iostream>
#include <cuda_runtime.h>

__global__ void vector_add_parallel(float *a, float *b, float *c, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

__global__ void matrix_add_scalar(float *mat, float scalar, float *out, int rows, int cols) {
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;


  if (row < rows && col < cols) {
    int idx = row * cols + col;
    out[idx] = mat[idx] + scalar;
  }
}

__global__ void matrix_add_scalar_series(float *mat, float scalar, float *out, int rows, int cols) {
  for (int i = 0; i < rows * cols; ++i) {
    out[i] = mat[i] + scalar;
  }
}

void checkCudaError(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

int main() {
  const int rows = 3;
  const int cols = 3;
  const int scalar = 10;

  float mat_cpp[rows * cols] = {1, 2, 3, 4, 5, 5, 7, 8, 9};

  float out_cpp[rows * cols] = {0};
  float* mat;
  float* out;

  checkCudaError(cudaMalloc((void**)&mat, rows * cols * sizeof(float)), "Alloc mat");
  checkCudaError(cudaMalloc((void**)&out, rows * cols * sizeof(float)), "Alloc out");

  checkCudaError(cudaMemcpy(mat, mat_cpp, rows * cols * sizeof(float), cudaMemcpyHostToDevice), "Memcpy mat");
  checkCudaError(cudaMemcpy(out, out_cpp, rows * cols * sizeof(float), cudaMemcpyHostToDevice), "Memcpy out");

  std::cout << "before run" << std::endl;

  dim3 blockSize(16, 16);
  dim3 gridSize(
      (cols + blockSize.x - 1) / blockSize.x,
      (rows + blockSize.y - 1) / blockSize.y
  );
  
  matrix_add_scalar<<<gridSize, blockSize>>>(mat, scalar, out, rows, cols);
  matrix_add_scalar_series<<<gridSize, blockSize>>>(mat, scalar, out, rows, cols);
  checkCudaError(cudaGetLastError(), "Kernel launch");
  checkCudaError(cudaDeviceSynchronize(), "Kernel execution");
  
  std::cout << "after run" << std::endl;

  checkCudaError(cudaMemcpy(out_cpp, out, rows * cols * sizeof(float), cudaMemcpyDeviceToHost), "Memcpy out back to out_cpp");

  std::cout << "[";
  for (int row = 0; row < rows; ++row) {
    std::cout << "[";
    for (int col = 0; col < cols; ++col) {
      std::cout << out_cpp[row * rows + col];
      if (col != cols-1) {
        std::cout << ", ";
      }
    }
    if (row != rows-1) {
      std::cout << "]\n";
    } else {
      std::cout << "]]\n";
    }
  }

  cudaFree(mat);
  cudaFree(out);
  return 0;
}