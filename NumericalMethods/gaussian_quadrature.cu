#include <cuda_runtime.h>
#include <iostream>

struct TwoPointGaussian {
  static constexpr double nodes[2] = {-0.57735026918962576463,
                                      0.57735026918962576463};
  static constexpr double weights[2] = {1.0, 1.0};
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


template <typename F>
__global__ void gaussian_quadrature(const F f, const unsigned int n1,
                                    const unsigned int n2, double *global_sum) {

  double2 range_x = calc_interval_pos(-1.0, 2, blockIdx.x, n1);
  double2 range_y = calc_interval_pos(-1.0, 2, blockIdx.y, n2);


  double v = rescale_interval(range_x, TwoPointGaussian::nodes[threadIdx.x]);
  double u = rescale_interval(range_y, TwoPointGaussian::nodes[threadIdx.y]);


}

int main(int argc, char *argv[]) {
  const unsigned int n1 = 10;
  const unsigned int n2 = 10;

  auto f = [] __device__(double x, double y) { return x * x + y * y; };
  double h_global_sum = 0.0;
  double *d_global_sum;

  dim3 block_size = {2, 2};
  dim3 grid_size = {n1, n2};

  // cuda malloc
  cudaMalloc(&d_global_sum, sizeof(double));
  cudaMemset(d_global_sum, 0, sizeof(double));

  gaussian_quadrature<<<grid_size, block_size>>>(f, n1, n2, d_global_sum);
  cudaDeviceSynchronize();

  cudaMemcpy(&h_global_sum, &d_global_sum, sizeof(double),
             cudaMemcpyDeviceToHost);
  //  cuda free
  cudaFree(d_global_sum);

  std::cout << "Result of the gaussian quadrature is : " << h_global_sum
            << std::endl;

  return 0;
}
