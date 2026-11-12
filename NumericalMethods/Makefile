build: gaussian_quadrature.cu
	nvcc -std=c++17 -arch=sm_60 --expt-extended-lambda gaussian_quadrature.cu -o gaussian_quadrature

run: build
	./gaussian_quadrature