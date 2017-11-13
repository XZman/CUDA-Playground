#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void reverseArrayKernal(int originalArr[], int reversedArr[], int len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len)
		reversedArr[len - i - 1] = originalArr[i];
}

void reverseArray(int originalArr[], int reversedArr[], int len) {
	int* d_arr;
	int* d_rev;

	cudaSetDevice(0);

	cudaMalloc(&d_arr, len * sizeof(int));
	cudaMalloc(&d_rev, len * sizeof(int));
	cudaMemcpy(d_arr, originalArr, len * sizeof(int), cudaMemcpyHostToDevice);

	reverseArrayKernal<<<len / 1024 + 1, 1024>>>(d_arr, d_rev, len);

	cudaMemcpy(reversedArr, d_rev, len * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_arr);
	cudaFree(d_rev);
}

int main() {
	const int len = 2147483647;
	int* arr = (int*)malloc(len * sizeof(int));
	int* reversedArr = (int*)malloc(len * sizeof(int));

	for (int i = 0; i < len; i++)
		arr[i] = i;

	time_t start, end;
	time(&start);

	reverseArray(arr, reversedArr, len);

	time(&end);
	double timedif = difftime(end, start);

	delete[] arr;

/*	for (int i = 0; i < len; i++)
		printf("%d ", reversedArr[i]);
	printf("\n");*/

	printf("Time elapsed: %lfs\n", timedif);

	delete[] reversedArr;

	system("pause");
	return 0;
}