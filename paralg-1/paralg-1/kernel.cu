
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

int main() {
	thrust::host_vector<int> data_h;
	thrust::device_vector<int> data_d;
	thrust::device_vector<int> out_d;

	for (int i = 0; i < 10; i++) {
		int x = i + ((i % 3) == 0) * 2;
		data_h.push_back(x);
		printf("%3d ", x);
	}
	putchar('\n');

	data_d = data_h;
	out_d.resize(data_d.size());

	thrust::exclusive_scan(data_d.begin(), data_d.end(), out_d.begin(), -1, thrust::maximum<int>());

	data_h = out_d;
	for (thrust::host_vector<int>::iterator i = data_h.begin(); i != data_h.end(); i++)
		printf("%3d ", *i);
	putchar('\n');
}