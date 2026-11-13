build: gaussian_quadrature.cu
	nvcc -std=c++20 -arch=sm_75  --expt-extended-lambda gaussian_quadrature.cu -o gaussian_quadrature

run: build
	./gaussian_quadrature