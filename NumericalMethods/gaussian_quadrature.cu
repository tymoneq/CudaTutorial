#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

#define N 16
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

template <typename F>
__global__ void
gaussian_quadratureWithReduction(const F f, const unsigned int n1,
                                 const unsigned int n2, double *global_sum,
                                 int *number_of_atomic_adds) {

  __shared__ double Sum[N * N];

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
  //  reduction
  unsigned int tid = threadIdx.y + (threadIdx.x * blockDim.y);
  Sum[tid] =
      sum * (1 / static_cast<double>(n1)) * (1 / static_cast<double>(n2));
  __syncthreads();

  for (unsigned int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
    if (tid < s) {
      Sum[tid] += Sum[tid + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    atomicAdd(global_sum, Sum[0]);
    atomicAdd(number_of_atomic_adds, 1);
  }
}

template <typename F>
__global__ void
gaussian_quadratureWithReductionWithoutAtomic(const F f, const unsigned int n1,
                                              const unsigned int n2,
                                              double *global_sum) {

  __shared__ double Sum[N * N];

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
  //  reduction
  unsigned int tid = threadIdx.y + (threadIdx.x * blockDim.y);
  Sum[tid] =
      sum * (1 / static_cast<double>(n1)) * (1 / static_cast<double>(n2));
  __syncthreads();

  for (unsigned int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
    if (tid < s) {
      Sum[tid] += Sum[tid + s];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    global_sum[blockIdx.y + blockIdx.x * blockDim.y] = Sum[0];
  }
}

__global__ void ReductionKernel(double *ReductionSum, const unsigned int n) {
  extern __shared__ double Sum[];
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  Sum[threadIdx.x] = ReductionSum[idx];
  Sum[threadIdx.x] = (idx < n) ? ReductionSum[idx] : 0.0;
  __syncthreads();

  const unsigned tid = threadIdx.x;
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      Sum[tid] += Sum[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    ReductionSum[blockIdx.x] = Sum[0];
  }
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

  const auto f = [] __device__(double x, double y) { return x * x; };
  const double result = PI / 4;
  double h_global_sum_slow = 0.0;
  double h_global_sum_fast = 0.0;
  double h_global_sum_reduction = 0.0;
  double *d_global_sum;

  // cuda malloc
  CUDA_CHECK(cudaMalloc(&d_global_sum, sizeof(double)));
  CUDA_CHECK(cudaMemset(d_global_sum, 0, sizeof(double)));

  // --- Configuration 1: Old 2x2 Block (1 interval per block, 4 threads/block)
  // #############################################################
  // #############################################################
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
  // #############################################################
  // #############################################################
  dim3 block_new(N, N);
  dim3 grid_new((n1 + block_new.x - 1) / block_new.x,
                (n2 + block_new.y - 1) / block_new.y);

  float time_new_ms = benchmark_kernel(
      [&]() {
        gaussian_quadrature<<<grid_new, block_new>>>(f, n1, n2, d_global_sum);
      },
      d_global_sum);

  cudaMemcpy(&h_global_sum_fast, d_global_sum, sizeof(double),
             cudaMemcpyDeviceToHost);

  // --- Configuration 3: New 16x16 Block with Reduction(256 intervals per
  // block, 256
  // #############################################################
  // #############################################################
  int h_number_of_atomics_adds = 0;
  int *d_number_of_atomics_adds;

  CUDA_CHECK(cudaMalloc(&d_number_of_atomics_adds, sizeof(int)));

  cudaMemset(d_number_of_atomics_adds, 0, sizeof(int));

  float time_new_ms_reduction = benchmark_kernel(
      [&]() {
        gaussian_quadratureWithReduction<<<grid_new, block_new>>>(
            f, n1, n2, d_global_sum, d_number_of_atomics_adds);
      },
      d_global_sum);

  cudaMemcpy(&h_global_sum_reduction, d_global_sum, sizeof(double),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(&h_number_of_atomics_adds, d_number_of_atomics_adds, sizeof(int),
             cudaMemcpyDeviceToHost);

  // --- Configuration 4: New 16x16 Block without atomic (256 intervals per
  // block, 256
  // #############################################################
  // #############################################################

  double h_ReductionSum = 0.0;
  double *d_ReductionSum;
  unsigned int size_reduction = n1 * n2;
  cudaMalloc(&d_ReductionSum, size_reduction);
  cudaMemset(&d_ReductionSum, 0.0, size_reduction);

  //  float time_new_ms_reduction_without_atomic = benchmark_kernel(
  //      [&]() {
  //        gaussian_quadratureWithReductionWithoutAtomic<<<grid_new,
  //        block_new>>>(
  //            f, n1, n2, d_global_sum);
  //        const int reduce_threads = 1024;

  //        while (size_reduction > 1) {
  //          int reduce_blocks =
  //              (size_reduction + reduce_threads - 1) / reduce_threads;
  //          size_t shared_mem = reduce_threads * sizeof(double);

  //          ReductionKernel<<<reduce_blocks, reduce_threads, shared_mem>>>(
  //              d_global_sum, size_reduction);

  //          size_reduction = reduce_blocks;
  //        }
  //      },
  //      d_global_sum);

  gaussian_quadratureWithReductionWithoutAtomic<<<grid_new, block_new>>>(
      f, n1, n2, d_ReductionSum);
  const int reduce_threads = 1024;

  cudaError_t err_launch = cudaGetLastError();
  if (err_launch != cudaSuccess) {
    std::cerr << "Błąd uruchomienia kernela: 1 " << cudaGetErrorString(err_launch)
              << std::endl;
  }

  while (size_reduction > 1) {
    int reduce_blocks = (size_reduction + reduce_threads - 1) / reduce_threads;
    size_t shared_mem = reduce_threads * sizeof(double);

    ReductionKernel<<<reduce_blocks, reduce_threads, shared_mem>>>(
        d_ReductionSum, size_reduction);
    cudaError_t err_launch = cudaGetLastError();
    if (err_launch != cudaSuccess) {
      std::cerr << "Błąd uruchomienia kernela: 2"
                << cudaGetErrorString(err_launch) << std::endl;
    }
    size_reduction = reduce_blocks;
  }
  cudaMemcpy(&h_ReductionSum, d_ReductionSum, sizeof(double),
             cudaMemcpyDeviceToHost);

  // --- Print Results -------------------------------------------
  // #############################################################
  // #############################################################
  // --- Print Results ---
  std::cout << std::fixed << std::setprecision(16);
  std::cout << "Result slow: " << h_global_sum_slow << "\n";
  std::cout << "Result fast: " << h_global_sum_fast << "\n";
  std::cout << "Result reduction: " << h_global_sum_reduction << "\n";
  std::cout << "Number of atomic adds in reduction: "
            << h_number_of_atomics_adds << "\n";
  std::cout << "Result reduction without atomic: " << h_ReductionSum << "\n";
  std::cout << "Error Slow: " << std::abs(h_global_sum_slow - result) << "\n";
  std::cout << "Error fast: " << std::abs(h_global_sum_fast - result) << "\n";
  std::cout << "Error reduction: " << std::abs(h_global_sum_reduction - result)
            << "\n";
  std::cout << "Error reduction without atomic: "
            << std::abs(h_ReductionSum - result) << "\n";
  std::cout << "Grid size: " << n1 << " x " << n2 << " (" << (n1 * n2)
            << " tiles)\n";
  std::cout << "--------------------------------------------------\n";
  std::cout << "Old (2x2 blocks):    " << time_old_ms << " ms\n";
  std::cout << "New (16x16 blocks):  " << time_new_ms << " ms\n";
  std::cout << "New Reduction (16x16 blocks):  " << time_new_ms_reduction
            << " ms\n";
  //  std::cout << "New Reduction without atomic(16x16 blocks):  "
  //            << time_new_ms_reduction_without_atomic << "ms\n";
  std::cout << "--------------------------------------------------\n";
  std::cout << "Speedup:             " << (time_old_ms / time_new_ms)
            << "x faster\n";
  std::cout << "Speedup:             " << (time_new_ms / time_new_ms_reduction)
            << "x faster\n";
  //  std::cout << "Total Speedup:             "
  //            << (time_old_ms / time_new_ms_reduction_without_atomic)
  //            << "x faster\n";

  cudaFree(d_ReductionSum);
  cudaFree(d_number_of_atomics_adds);
  cudaFree(d_global_sum);
  return 0;
}
