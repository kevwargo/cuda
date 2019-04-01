
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include <ctime>


__global__ void movingAverage(int size, int d, int *data)
{
	extern __shared__ int smem[];
	int btid = threadIdx.x;
	int gtid = btid + blockIdx.x*blockDim.x;
	int idx1 = gtid - (d - 1) / 2;
	int idx2 = blockDim.x + gtid - (d - 1) / 2;
	
	if (idx1 > size) {
		return;
	}
	if (idx2 > size) {
		return;
	}

	smem[btid] = data[idx1];
	if (btid < d - 1) {
		smem[blockDim.x + btid] = data[idx2];
	}

}

int main()
{
	int bs = 512;
	int gs = 32;
	int size = bs * gs;
	int d = 7;
	thrust::host_vector<int> data_h;
	thrust::device_vector<int> data_d;
	int *movavg = new int[size];
	srand(time(NULL));

	for (int i = 0; i < size; i++) {
		int x = rand() % 100;
		data_h.push_back(x);
	}
	for (int i = 0; i < size; i++) {
		int sum = 0;
		int start = i - (d - 1) / 2;
		int end = i + (d - 1) / 2;
		while (start < 0) {
			sum += data_h[0];
			start++;
		}
		while (end >= size) {
			sum += data_h[size - 1];
			end--;
		}

		for (int j = start; j <= end; j++) {
			sum += data_h[j];
		}
		movavg[i] = sum / d;
	}
	
	data_d = data_h;
	movingAverage << <gs, bs, sizeof(int)*(bs + d - 1) >> > (size, d, data_d.data().get());
	data_h = data_d;

	int errcount = 0;
	for (int i = 0; i < size; i++) {
		if (data_h[i] != movavg[i]) {
			errcount++;
		}
	}

	printf("Error count: %d\n", errcount);
}
