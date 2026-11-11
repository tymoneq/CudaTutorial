#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

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
__global__ void gaussian_quadrature(const F f, const unsigned int n1,
                                    const unsigned int n2, double *global_sum) {

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

int main(int argc, char *argv[]) {
  const unsigned int n1 = 10;
  const unsigned int n2 = 10;

  auto f = [] __device__(double x, double y) { return x * x ; };
  double h_global_sum = 0.0;
  double *d_global_sum;

  dim3 block_size = {2, 2};
  dim3 grid_size = {n1, n2};

  // cuda malloc
  CUDA_CHECK(cudaMalloc(&d_global_sum, sizeof(double)));
  CUDA_CHECK(cudaMemset(d_global_sum, 0, sizeof(double)));

  gaussian_quadrature<<<grid_size, block_size>>>(f, n1, n2, d_global_sum);
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(&h_global_sum, d_global_sum, sizeof(double),
                        cudaMemcpyDeviceToHost));
  //  cuda free
  cudaFree(d_global_sum);

  std::cout << "Result of the gaussian quadrature is : " << std::fixed << std::setprecision(10)<< h_global_sum
            << std::endl;

  return 0;
}
