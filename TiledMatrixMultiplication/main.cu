#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <stdio.h>

constexpr int M = 1 << 15;
constexpr int BLOCKSIZE = 16;

__global__ void FastMatrix(float *A, float *B, float *C, const int N) {

  int by = blockIdx.y;
  int bx = blockIdx.x;

  int ty = threadIdx.y;
  int tx = threadIdx.x;

  int i = blockDim.y * by + ty;
  int j = blockDim.x * bx + tx;

  //  allocating shared memory

  __shared__ float sh_A[BLOCKSIZE][BLOCKSIZE];
  __shared__ float sh_B[BLOCKSIZE][BLOCKSIZE];

  float value = 0;
  int num_tiles = (N + BLOCKSIZE - 1) / BLOCKSIZE;

  for (int phase = 0; phase < num_tiles; phase++) {

    if (i < N && (phase * BLOCKSIZE + tx) < N)
      sh_A[ty][tx] = A[(i)*N + phase * BLOCKSIZE + tx];
    else
      sh_A[ty][tx] = 0;
    if ((phase * BLOCKSIZE + ty) < N && j < N)
      sh_B[ty][tx] = B[(phase * BLOCKSIZE + ty) * N + j];
    else
      sh_B[ty][tx] = 0;
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCKSIZE; k++)
      value += sh_A[ty][k] * sh_B[k][tx];
    __syncthreads();
  }
  if (i < N && j < N)
    C[i * N + j] = value;
}

void initMatric(float *Matrix, int row, int col) {

  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++)
      Matrix[i * row + j] = (float)rand() / (float)RAND_MAX;
}

int main() {

  srand(time(NULL));

  float *A = new float[M * M];
  float *B = new float[M * M];
  float *C = new float[M * M];

  float *d_A;
  float *d_B;
  float *d_C;

  size_t size = M * M * sizeof(float);

  initMatric(A, M, M);
  initMatric(B, M, M);

  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

  dim3 blockdim(BLOCKSIZE, BLOCKSIZE);
  dim3 gridDim(((M + BLOCKSIZE - 1) / BLOCKSIZE),
               ((M + BLOCKSIZE - 1) / BLOCKSIZE));

  FastMatrix<<<gridDim, blockdim>>>(d_A, d_B, d_C, M);
  cudaDeviceSynchronize();

  //  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

  //  for (int i = 0; i < M; i++) {
  //    for (int j = 0; j < M; j++)
  //      printf("%f ", C[i * M + j]);
  //    printf("\n");
  //  }

  //  free memory
  delete[] A;
  delete[] B;
  delete[] C;
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
