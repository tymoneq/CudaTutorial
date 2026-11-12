#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

// more blocks but worst precision

#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    cudaError_t error = (err);                                                 \
    if (error != cudaSuccess) {                                                \
      printf("CUDA Błąd: %s w pliku %s:%d\n", cudaGetErrorString(error),       \
             __FILE__, __LINE__);                                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

constexpr double PI = 3.141592653589793238462643383279502884;

struct TwoPointGaussian {
  __host__ __device__ static constexpr double node(int i) {
    return (i == 0) ? -0.57735026918962576463 : 0.57735026918962576463;
  }
};

__device__ double2 calc_interval_pos(double start, double len, int current,
                                     double total_num_of_blocks) {
  double x = start + (len / total_num_of_blocks) * current;
  double y = start + (len / total_num_of_blocks) * (current + 1);

  return make_double2(x, y);
}

__device__ double rescale_interval(const double2 range, const double value) {
  return (range.y - range.x) / 2 * value + (range.x + range.y) / 2;
}

__device__ double2 calc_polar(const double u, const double v) {

  double r = (u + 1) / 2;
  double alfa = (v + 1) * PI;

  return make_double2(r, alfa);
}

__device__ double calc_jacobian(const double u) { return PI * (u + 1) / 4; }

__device__ double2 to_cartesian(double2 polar) {
  double sin_a, cos_a;
  sincos(polar.y, &sin_a, &cos_a);

  double x = polar.x * cos_a;
  double y = polar.x * sin_a;

  return make_double2(x, y);
}

template <typename F>
__global__ void gaussian_quadratureSlow(const F f, const unsigned int n1,
                                        const unsigned int n2,
                                        double *global_sum) {

  double2 range_x = calc_interval_pos(-1.0, 2.0, blockIdx.x, n1);
  double2 range_y = calc_interval_pos(-1.0, 2.0, blockIdx.y, n2);

  double u = rescale_interval(range_x, TwoPointGaussian::node(threadIdx.x));
  double v = rescale_interval(range_y, TwoPointGaussian::node(threadIdx.y));

  double2 polar = calc_polar(u, v);
  double Jacobian = calc_jacobian(u);

  double2 cartesian = to_cartesian(polar);

  double val = Jacobian * (1 / static_cast<double>(n1)) *
               (1 / static_cast<double>(n2)) * f(cartesian.x, cartesian.y);

  atomicAdd(global_sum, val);
}

template <typename F>
__global__ void gaussian_quadrature(const F f, const unsigned int n1,
                                    const unsigned int n2, double *global_sum) {

  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx >= n1 || idy >= n2) {
    return;
  }

  double sum = 0.0;

  double2 range_x = calc_interval_pos(-1.0, 2.0, idx, n1);
  double2 range_y = calc_interval_pos(-1.0, 2.0, idy, n2);

  for (int i = 0; i < 2; i++) {
    double u = rescale_interval(range_x, TwoPointGaussian::node(i));
    double Jacobian = calc_jacobian(u);
    for (int j = 0; j < 2; j++) {

      double v = rescale_interval(range_y, TwoPointGaussian::node(j));

      double2 polar = calc_polar(u, v);

      double2 cartesian = to_cartesian(polar);

      sum += Jacobian * f(cartesian.x, cartesian.y);
    }
  }
  double val =
      sum * (1 / static_cast<double>(n1)) * (1 / static_cast<double>(n2));

  atomicAdd(global_sum, val);
}

template <typename KernelLauncher>
float benchmark_kernel(KernelLauncher launch, double *d_sum,
                       int warmup_runs = 5, int timed_runs = 25) {
  // Warm-up runs (not timed)
  for (int i = 0; i < warmup_runs; ++i) {
    cudaMemset(d_sum, 0, sizeof(double));
    launch();
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < timed_runs; i++) {
    cudaMemset(d_sum, 0, sizeof(double));
    launch();
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float total_ms = 0.0f;
  cudaEventElapsedTime(&total_ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return total_ms / timed_runs; // Average time per kernel execution (ms)
}

int main(int argc, char *argv[]) {
  const unsigned int n1 = 2048;
  const unsigned int n2 = 2048;
  const unsigned int n = 16;

  auto f = [] __device__(double x, double y) { return x * x + y * y; };
  double h_global_sum_slow = 0.0;
  double h_global_sum_fast = 0.0;
  double *d_global_sum;

  // cuda malloc
  CUDA_CHECK(cudaMalloc(&d_global_sum, sizeof(double)));
  CUDA_CHECK(cudaMemset(d_global_sum, 0, sizeof(double)));

  // --- Configuration 1: Old 2x2 Block (1 interval per block, 4 threads/block)
  // ---
  dim3 block_old(2, 2);
  dim3 grid_old(n1, n2);

  float time_old_ms = benchmark_kernel(
      [&]() {
        gaussian_quadratureSlow<<<grid_old, block_old>>>(f, n1, n2,
                                                         d_global_sum);
      },
      d_global_sum);

  cudaMemcpy(&h_global_sum_slow, d_global_sum, sizeof(double),
             cudaMemcpyDeviceToHost);
  // --- Configuration 2: New 16x16 Block (256 intervals per block, 256
  // threads/block) ---
  dim3 block_new(n, n);
  dim3 grid_new((n1 + block_new.x - 1) / block_new.x,
                (n2 + block_new.y - 1) / block_new.y);

  float time_new_ms = benchmark_kernel(
      [&]() {
        gaussian_quadrature<<<grid_new, block_new>>>(f, n1, n2, d_global_sum);
      },
      d_global_sum);

  cudaMemcpy(&h_global_sum_fast, d_global_sum, sizeof(double),
             cudaMemcpyDeviceToHost);

  double err = std::abs(h_global_sum_fast - h_global_sum_slow);

  // --- Print Results ---
  std::cout << std::fixed << std::setprecision(16);
  std::cout << "Result slow: " << h_global_sum_slow << "\n";
  std::cout << "Result fast: " << h_global_sum_fast << "\n";
  std::cout << "Error: " << err << "\n";
  std::cout << "Grid size: " << n1 << " x " << n2 << " (" << (n1 * n2)
            << " tiles)\n";
  std::cout << "--------------------------------------------------\n";
  std::cout << "Old (2x2 blocks):    " << time_old_ms << " ms\n";
  std::cout << "New (16x16 blocks):  " << time_new_ms << " ms\n";
  std::cout << "Speedup:             " << (time_old_ms / time_new_ms)
            << "x faster\n";
  return 0;
}
