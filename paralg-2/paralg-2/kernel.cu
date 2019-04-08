
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

struct bitCount : public thrust::binary_function<int, int, int> {
	// doesn't work because it's not associative
	__device__ int operator()(int x, int y) {
		return x + __popc(y);
	}
};

int main() {
	thrust::host_vector<int> src_h;
	thrust::device_vector<int> src_d;

	for (int i = 0; i < 10; i++) {
		src_h.push_back(i);
	}

	src_d = src_h;

	int bit_sum = thrust::reduce(src_d.begin(), src_d.end(), 0, bitCount());

	printf("%d\n", bit_sum);
}