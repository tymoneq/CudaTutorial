#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10'000'000
#define BLOCK_SIZE 256

void add_vector_cpu(float *a, float *b, float *c, int n) {
  for (int i = 0; i < n; i++)
    c[i] = a[i] + b[i];
}

__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

void init_vector(float *vec, int n) {
  for (int i = 0; i < n; i++)
    vec[i] = (float)rand() / RAND_MAX;
}

double get_time() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
  float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
  float *d_a, *d_b, *d_c;

  size_t size = N * sizeof(float);

  h_a = (float *)malloc(size);
  h_b = (float *)malloc(size);
  h_c_cpu = (float *)malloc(size);
  h_c_gpu = (float *)malloc(size);

  srand(time(NULL));
  init_vector(h_a, N);
  init_vector(h_b, N);

  cudaMalloc(&d_a, N);
  cudaMalloc(&d_b, N);
  cudaMalloc(&d_c, N);

  cudaMemcpy(d_a, h_a, N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N, cudaMemcpyHostToDevice);

  int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  printf("Performing warm-up runs\n");

  for (int i = 0; i < 3; i++) {
    add_vector_cpu(h_a, h_b, h_c_cpu, N);
    vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
  }

  printf("Benchmarking CPU implementation ...\n");

  double cpu_total_time = 0.0;
  for (int i = 0; i < 20; i++) {
    double start = get_time();
    add_vector_cpu(h_a, h_b, h_c_cpu, N);
    double end_time = get_time();
    cpu_total_time += end_time - start;
  }

  double avg_cpu_time = cpu_total_time / 20;

  printf("Benchmarking GPU implementation\n");

  double gpu_total_time = 0.0;
  for (int i = 0; i < 20; i++) {
    double start = get_time();
    vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    double end_time = get_time();
    gpu_total_time += end_time - start;
  }

  double avg_gpu_time = gpu_total_time / 20;

  printf("CPU average time: %f milliseconds\n", avg_cpu_time * 1000);
  printf("GPU average time: %f milliseconds\n", avg_gpu_time * 1000);
  printf("Speedup: %fx\n", avg_cpu_time / avg_gpu_time);

  // Verify results (optional)
  cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
  bool correct = true;
  for (int i = 0; i < N; i++) {
    if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-10) {
      correct = false;
      break;
    }
  }
  printf("Results are %s\n", correct ? "correct" : "incorrect");

  // Free memory
  free(h_a);
  free(h_b);
  free(h_c_cpu);
  free(h_c_gpu);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}