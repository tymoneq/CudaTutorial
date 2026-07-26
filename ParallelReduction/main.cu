#include <cstddef>
#include <cuda_runtime.h>
#include <stdio.h>

constexpr int N = 100'000'000;
constexpr int BLOCK_SIZE = 256;

__global__ void ParallelReduction(float *arr_in, float *global_sum, int n) {

  extern __shared__ float sdata[];

  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  sdata[tid] = arr_in[myId];
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && myId < n) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0)
    atomicAdd(global_sum, sdata[0]);
}

void initArr(float *arr, int n) {
  for (int i = 0; i < n; i++)
    arr[i] = (float)rand() / (float)RAND_MAX;
}

int main() {
  float *arr = new float[N];
  float host_sum = 0;
  size_t size = N * sizeof(float);
  float *d_arr, *d_sum;

  initArr(arr, N);
  cudaMalloc(&d_arr, size);
  cudaMalloc(&d_sum, 4);
  cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

  dim3 gridDim = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  ParallelReduction<<<gridDim, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
      d_arr, d_sum, N);
  cudaDeviceSynchronize();

  cudaMemcpy(&host_sum, d_sum, 4, cudaMemcpyDeviceToHost);

  printf("%f\n", host_sum);

  delete[] arr;
  cudaFree(d_arr);
  cudaFree(d_sum);

  return 0;
}
