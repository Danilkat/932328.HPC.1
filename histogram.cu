#include <iostream>
#include <cstdlib> // For rand() function
#include <ctime>   // For time() function
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cassert>

#define BIN_COUNT 256
#define SHARED_BIN_COUNT 32

__shared__ unsigned int d_bin_data_shared[BIN_COUNT];

unsigned char* generateRandomString(int size) {
    std::srand(std::time(0));
    size_t dataSize = size * sizeof(unsigned char);

    unsigned char* randomString = (unsigned char*)malloc(dataSize);

    for (int i = 0; i < size; ++i) {
        unsigned char randomChar = (unsigned char)(std::rand() % BIN_COUNT);
        //unsigned char randomChar = (unsigned char)(32);
        randomString[i] = randomChar;
    }

    return randomString;
}

/* Each read is 4 bytes, not one, 32 x 4 =  128 byte reads */
/* Accumulate into shared memory N times */
__global__ void histogramKernel(unsigned int* d_hist_data,
    unsigned int* d_bin_data)
{
    /* Work out our thread id */
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int tid =  idx + idy * blockDim.x * gridDim.x;

    /* Clear shared memory */
    d_bin_data_shared[threadIdx.x] = 0;

    /* Wait for all threads to update shared memory */
    __syncthreads();

    unsigned int value_u32 =  d_hist_data[tid];
    
    //atomicAdd(&(d_bin_data_shared[0]), 1);
    atomicAdd(&(d_bin_data_shared[((value_u32 & 0x000000FF))]), 1);
    atomicAdd(&(d_bin_data_shared[((value_u32 & 0x0000FF00) >> 8)]), 1);
    atomicAdd(&(d_bin_data_shared[((value_u32 & 0x00FF0000) >> 16)]), 1);
    atomicAdd(&(d_bin_data_shared[((value_u32 & 0xFF000000) >> 24)]), 1);

    /* Wait for all threads to update shared memory */
    __syncthreads();

    /* The write the accumulated data back to global memory in blocks, not scattered */
    atomicAdd(&(d_bin_data[threadIdx.x]), d_bin_data_shared[threadIdx.x]);
}

void myhistogramCPU(unsigned int* h_hist_data, unsigned int* h_bin_data, unsigned int N) {
    unsigned int start_time = clock();
    printf("%d\n", sizeof((unsigned int*)h_hist_data)[0]);
    printf("%d\n", sizeof(h_hist_data[0]));

    for (unsigned int i = 0; i < BIN_COUNT; i++) h_bin_data[i] = 0;
    for (unsigned int i = 0; i < N / 4; i++) {
        unsigned int data = ((unsigned int*)h_hist_data)[i];
        h_bin_data[(data & 0x000000FF)]++;
        h_bin_data[((data & 0x0000FF00) >> 8)]++;
        h_bin_data[((data & 0x00FF0000) >> 16)]++;
        h_bin_data[((data & 0xFF000000) >> 24)]++;
    }

    unsigned int end_time = clock();
    unsigned int search_time = end_time - start_time;
    printf("Время выполнения CPU: %d мс.\n", search_time);
}

cudaError_t histogramWithCuda(unsigned int* h_hist_data, unsigned int* h_bin_data_GPU, unsigned int size) {
    printf("%d\n", sizeof(((unsigned int*)h_hist_data)[0]));
    printf("%d\n", sizeof(h_hist_data[0]));
    unsigned int* d_hist_data;
    unsigned int* d_bin_data;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_hist_data, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_bin_data, BIN_COUNT * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Высчитываем с момента копирования
    unsigned int start_time = clock();

    cudaStatus = cudaMemcpy(d_hist_data, h_hist_data, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    histogramKernel << <size / 128,
        32 >> > (
            d_hist_data, d_bin_data);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "histogramKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching histogramKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(h_bin_data_GPU, d_bin_data, BIN_COUNT * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    unsigned int end_time = clock();
    unsigned int search_time = end_time - start_time;
    printf("Время выполнения GPU: %d мс.", search_time);


Error:
    cudaFree(d_hist_data);
    cudaFree(d_bin_data);

    return cudaStatus;
}

int histogramWrapper(unsigned int size) {
    unsigned char* h_hist_data;
    unsigned int* h_bin_data_CPU, * h_bin_data_GPU;

    assert((size % 128u) == 0u);

    h_hist_data = generateRandomString(size);
    printf("%d\n", sizeof(h_hist_data));
    /*for (int i = 0; i < size; i++)
    {
        if ((int)h_hist_data[i] > 31) {
            std::cout << h_hist_data[i];
        }
        std::cout << (int)h_hist_data[i];
        std::cout << std::endl;
    }*/
    h_bin_data_CPU = (unsigned int*)malloc(BIN_COUNT * sizeof(unsigned int));
    h_bin_data_GPU = (unsigned int*)malloc(BIN_COUNT * sizeof(unsigned int));

    myhistogramCPU((unsigned int*)h_hist_data, h_bin_data_CPU, size);

    //std::cout << "\n0 - " << h_bin_data_CPU[0] << std::endl;
    /*for (int i = 0; i < BIN_COUNT; i++)
    {
        if (h_bin_data_CPU[i] > 0) {
            std::cout << i << " - " << h_bin_data_CPU[i] << std::endl;
        }
    }
    std::cout << std::endl;*/

    cudaError_t cudaStatus = histogramWithCuda((unsigned int*)h_hist_data, h_bin_data_GPU, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "histogramWithCuda failed!");
        return 1;
    }
    std::cout << "\nСравнение результатов...\n";
    bool match = true;
    for (size_t i = 0; i < BIN_COUNT; i++)
    {
        printf("Index %d. Asserted: %d. From kernel: %d ", i, h_bin_data_CPU[i], h_bin_data_GPU[i]);
        if (h_bin_data_CPU[i] != h_bin_data_GPU[i]) {
            printf("FAILED \n");
            match = false;
        }
        else {
            printf("\n");
        }
    }
    std::cout << (match ? "Результаты сошлись" : "Результаты не сошлись");

    //std::cout << "\nGPU:\n";
    //std::cout << 0 << " - " << h_bin_data_GPU[0] << std::endl;
    /*for (int i = 0; i < BIN_COUNT; i++)
    {
        if (h_bin_data_GPU[i] > 0) {
            std::cout << i << " - " << h_bin_data_GPU[i] << std::endl;
        }
    }*/


    return 0;

}