#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

#define N 16
// more blocks but worst precision

#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    cudaError_t error = (err);                                                 \
    if (error != cudaSuccess) {                                                \
      printf("CUDA Error: %s w pliku %s:%d\n", cudaGetErrorString(error),      \
             __FILE__, __LINE__);                                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

constexpr double PI = 3.1415926535897932;

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
                                 const unsigned int n2, double *global_sum) {

  __shared__ double Sum[N * N];

  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
  double sum = 0.0;
  if (idx < n1 && idy < n2) {

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
  double sum = 0.0;
  if (idx < n1 && idy < n2) {
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
    unsigned int block_id = blockIdx.y * gridDim.x + blockIdx.x;
    global_sum[block_id] = Sum[0];
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

template <typename T> struct CudaTestCase {
  T func;
  double exactVal;
  std::string name;
};

template <typename F> CudaTestCase(F, double, const char *) -> CudaTestCase<F>;

template <typename Tuple, typename Func>
void for_each_in_tuple(const Tuple &t, Func &&f) {
  std::apply([&](const auto &...args) { (f(args), ...); }, t);
}

struct KernelBenchmark {
  std::string kernel_name;
  double result;
  double error;
};

void print_console_table(const std::string &func_name, unsigned int n1,
                         unsigned int n2,
                         const std::vector<KernelBenchmark> &results) {
  unsigned long long total_points = 4ULL * n1 * n2;

  std::cout << "\n============================================================="
               "===================\n";
  std::cout << " Funkcja: " << func_name << " | Siatka: " << n1 << "x" << n2
            << " | Punkty Gaussa: " << total_points << "\n";
  std::cout << "==============================================================="
               "=================\n";

  std::cout << std::left << std::setw(30) << "Wariant kernela" << std::right
            << std::setw(22) << "Wynik" << std::setw(16) << "Blad bezwzg."
            << "\n";
  std::cout << "---------------------------------------------------------------"
               "-----------------\n";

  for (const auto &r : results) {
    std::cout << std::left << std::setw(30) << r.kernel_name << std::right
              << std::fixed << std::setprecision(16) << std::setw(22)
              << r.result << std::scientific << std::setprecision(2)
              << std::setw(16) << r.error << std::fixed << std::setprecision(4)
              << "\n";
  }
  std::cout << "==============================================================="
               "=================\n";
}
void export_to_csv(const std::string &filename, const std::string &func_name,
                   unsigned int n1, unsigned int n2,
                   const std::vector<KernelBenchmark> &results,
                   bool write_header = false) {
  // Otwieramy w trybie dopisywania (std::ios::app), żeby zbierać kolejne testy
  // w jednym pliku
  std::ofstream file(filename, std::ios::app);
  if (!file.is_open())
    return;

  if (write_header) {
    file << "Function,Grid_N1,Grid_N2,Total_Points,Kernel,Result,Abs_Error\n";
  }

  unsigned long long total_points = 4ULL * n1 * n2;

  for (const auto &r : results) {
    file << "\"" << func_name << "\"," << n1 << "," << n2 << "," << total_points
         << ","
         << "\"" << r.kernel_name
         << "\","
         // setprecision(17) gwarantuje zapis pełnej precyzji IEEE-754 double
         // bez utraty bitów
         << std::fixed << std::setprecision(17) << r.result << ","
         << std::scientific << std::setprecision(6) << r.error << ","
         << "\n";
  }
}

int main(int argc, char *argv[]) {
  const auto f1 = [] __device__(double x, double y) -> double { return 1.0; };
  const auto f2 = [] __device__(double x, double y) -> double {
    return x * x + y * y;
  };
  const auto f3 = [] __device__(double x, double y) -> double { return x * x; };
  const auto f4 = [] __device__(double x, double y) -> double { return x; };
  const auto f5 = [] __device__(double x, double y) -> double {
    double r2 = x * x + y * y;
    return r2 * r2;
  };
  const int reduce_threads = 1024;
  // std::make_tuple przechowuje 5 różnych typów CudaTestCase<decltype(f_i)>
  auto allFunctions =
      std::make_tuple(CudaTestCase{f1, PI, "f(x,y) = 1"},
                      CudaTestCase{f2, PI / 2.0, "f(x,y) = x^2 + y^2"},
                      CudaTestCase{f3, PI / 4.0, "f(x,y) = x^2"},
                      CudaTestCase{f4, 0.0, "f(x,y) = x"},
                      CudaTestCase{f5, PI / 3.0, "f(x,y) = (x^2 + y^2)^2"});

  std::cout << std::fixed << std::setprecision(16);

  bool is_first_row = true;
  std::ofstream("benchmark_results.csv", std::ios::trunc).close();

  for_each_in_tuple(
      allFunctions,
      [&](const auto &test) {
        std::cout << test.name << "\n";

        for (int block_size = 512; block_size <= 4096; block_size *= 2) {

          double h_global_sum_slow = 0.0;
          double h_global_sum_fast = 0.0;
          double h_global_sum_reduction = 0.0;
          double *d_global_sum;
          CUDA_CHECK(cudaMalloc(&d_global_sum, sizeof(double)));
          CUDA_CHECK(cudaMemset(d_global_sum, 0, sizeof(double)));

          // Kernel 1
          // #############################################################
          // #############################################################
          dim3 block_old(2, 2);
          dim3 grid_old(block_size, block_size);

          gaussian_quadratureSlow<<<grid_old, block_old>>>(
              test.func, block_size, block_size, d_global_sum);

          cudaMemcpy(&h_global_sum_slow, d_global_sum, sizeof(double),
                     cudaMemcpyDeviceToHost);

          // kernel 2
          // #############################################################
          // #############################################################
          CUDA_CHECK(cudaMemset(d_global_sum, 0, sizeof(double)));
          dim3 block_new(N, N);
          dim3 grid_new((block_size + block_new.x - 1) / block_new.x,
                        (block_size + block_new.y - 1) / block_new.y);

          gaussian_quadrature<<<grid_new, block_new>>>(
              test.func, block_size, block_size, d_global_sum);
          cudaMemcpy(&h_global_sum_fast, d_global_sum, sizeof(double),
                     cudaMemcpyDeviceToHost);

          // kernel 3
          // #############################################################
          // #############################################################
          CUDA_CHECK(cudaMemset(d_global_sum, 0, sizeof(double)));
          gaussian_quadratureWithReduction<<<grid_new, block_new>>>(
              test.func, block_size, block_size, d_global_sum);
          cudaMemcpy(&h_global_sum_reduction, d_global_sum, sizeof(double),
                     cudaMemcpyDeviceToHost);

          // kernel 4
          // #############################################################
          // #############################################################
          double h_ReductionSum = 0.0;
          double *d_ReductionSum;
          unsigned int size_reduction = block_size * block_size;
          CUDA_CHECK(
              cudaMalloc(&d_ReductionSum, size_reduction * sizeof(double)));
          CUDA_CHECK(
              cudaMemset(d_ReductionSum, 0.0, size_reduction * sizeof(double)));

          gaussian_quadratureWithReductionWithoutAtomic<<<grid_new,
                                                          block_new>>>(
              test.func, block_size, block_size, d_ReductionSum);

          unsigned int local_size_reduction = size_reduction;
          while (local_size_reduction > 1) {
            int reduce_blocks =
                (local_size_reduction + reduce_threads - 1) / reduce_threads;
            size_t shared_mem = reduce_threads * sizeof(double);

            ReductionKernel<<<reduce_blocks, reduce_threads, shared_mem>>>(
                d_ReductionSum, local_size_reduction);

            local_size_reduction = reduce_blocks;
          }
          cudaMemcpy(&h_ReductionSum, d_ReductionSum, sizeof(double),
                     cudaMemcpyDeviceToHost);

          // --- Print Results -------------------------------------------
          // #############################################################
          // #############################################################
          // --- Print Results ---

          std::vector<KernelBenchmark> kernelBenchmark = {
              {"slow", h_global_sum_slow,
               std::abs(h_global_sum_slow - test.exactVal)},
              {"fast", h_global_sum_fast,
               std::abs(h_global_sum_fast - test.exactVal)},
              {"reduction with atomic", h_global_sum_reduction,
               std::abs(test.exactVal - h_global_sum_reduction)},
              {"reduction without atomic", h_ReductionSum,
               std::abs(h_ReductionSum - test.exactVal)}};

          print_console_table(test.name, block_size, block_size,
                              kernelBenchmark);
          export_to_csv("benchmark_results.csv", test.name, block_size,
                        block_size, kernelBenchmark, is_first_row);
          is_first_row = false;
          cudaFree(d_ReductionSum);
          cudaFree(d_global_sum);
        }
      }

  );

  return 0;
}
