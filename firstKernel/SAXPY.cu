#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

#define N 10'000'000
#define BLOCK_SIZE 256
#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    cudaError_t error = (err);                                                 \
    if (error != cudaSuccess) {                                                \
      printf("CUDA Błąd: %s w pliku %s:%d\n", cudaGetErrorString(error),       \
             __FILE__, __LINE__);                                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void SAXPY(float *Scalar, float *A, float *B, float *C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    C[i] = Scalar[i] * A[i] + B[i];
}

void init_vec(float *vec, size_t size) {
  for (size_t i = 0; i < size; i++)
    vec[i] = i;
}

int main() {
  int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  srand(time(NULL));

  float *h_scalar, *h_A, *h_B, *h_C;
  float *d_scalar, *d_A, *d_B, *d_C;

  size_t size = N * sizeof(float);

  h_scalar = (float *)malloc(size);
  h_A = (float *)malloc(size);
  h_B = (float *)malloc(size);
  h_C = (float *)malloc(size);

  init_vec(h_scalar, N);
  init_vec(h_A, N);
  init_vec(h_B, N);

  printf("%f\n", h_scalar[5]);
  printf("%f\n", h_A[5]);
  printf("%f\n", h_B[5]);

  CUDA_CHECK(cudaMalloc(&d_scalar, size));
  CUDA_CHECK(cudaMalloc(&d_A, size));
  CUDA_CHECK(cudaMalloc(&d_B, size));
  CUDA_CHECK(cudaMalloc(&d_C, size));

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scalar, h_scalar, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  SAXPY<<<num_blocks, BLOCK_SIZE>>>(d_scalar, d_A, d_B, d_C, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  printf("%f\n", h_C[5]);

  free(h_scalar);
  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_scalar);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
